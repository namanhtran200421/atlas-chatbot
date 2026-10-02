/**
 * Local SigV4 signing proxy for the Membership Atlas RAG API.
 *
 * `POST /chat` on the API Gateway HTTP API is protected with AWS_IAM, so every
 * request must be SigV4-signed. A browser cannot do that without credentials,
 * so during development the Angular dev server proxies /api/* to this process,
 * which signs with your local AWS credential chain (env vars, shared profile,
 * or SSO) and forwards the request.
 *
 *   browser -> /api/chat -> ng serve proxy -> this -> API Gateway (AWS_IAM)
 *
 * Nothing here ships to the browser. Run `npm start` to launch it alongside
 * `ng serve`, or `npm run proxy` on its own.
 */

import { createServer } from 'node:http';
import { request as httpsRequest } from 'node:https';

import { fromNodeProviderChain } from '@aws-sdk/credential-providers';
import aws4 from 'aws4';

const PORT = Number(process.env.PROXY_PORT ?? 4300);
const REGION = process.env.AWS_REGION ?? 'ap-southeast-2';
const UPSTREAM_HOST =
  process.env.MEMBERSHIP_RAG_HOST ?? 'may54zk1a5.execute-api.ap-southeast-2.amazonaws.com';
const SERVICE = 'execute-api';
const AUTH_MODE = (process.env.CHAT_AUTH_MODE ?? 'cognito').toLowerCase();
const MAX_BODY_BYTES = 64 * 1024;

if (!['iam', 'cognito'].includes(AUTH_MODE)) {
  throw new Error('CHAT_AUTH_MODE must be either "iam" or "cognito".');
}

const loadCredentials = fromNodeProviderChain({ profile: process.env.AWS_PROFILE });
let cached = null;

/** Credentials from the standard chain, refreshed a minute before they lapse. */
async function credentials() {
  const soon = Date.now() + 60_000;
  if (cached && (!cached.expiration || cached.expiration.getTime() > soon)) return cached;
  cached = await loadCredentials();
  return cached;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (chunk) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error('request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function send(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Cache-Control': 'no-store',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

/** Sign the request and pipe the upstream response straight back. */
function forward(res, signed, body) {
  return new Promise((resolve) => {
    const upstream = httpsRequest(
      {
        host: signed.host ?? signed.hostname,
        path: signed.path,
        method: signed.method,
        headers: signed.headers,
      },
      (response) => {
        const chunks = [];
        response.on('data', (chunk) => chunks.push(chunk));
        response.on('end', () => {
          const payload = Buffer.concat(chunks);
          res.writeHead(response.statusCode ?? 502, {
            'Content-Type': response.headers['content-type'] ?? 'application/json',
            'Cache-Control': 'no-store',
            'Content-Length': payload.length,
          });
          res.end(payload);
          resolve(response.statusCode ?? 502);
        });
      },
    );

    upstream.on('error', (error) => {
      send(res, 502, { error: 'upstream_unreachable', message: error.message });
      resolve(502);
    });

    if (body) upstream.write(body);
    upstream.end();
  });
}

const server = createServer(async (req, res) => {
  const started = Date.now();
  // The dev server forwards the /api prefix untouched; strip it here.
  const path = (req.url ?? '/').replace(/^\/api/, '') || '/';

  if (req.method === 'OPTIONS') {
    res.writeHead(204).end();
    return;
  }

  if (path === '/health' || path === '/') {
    send(res, 200, { service: 'membership-rag-signing-proxy', upstream: UPSTREAM_HOST });
    return;
  }

  if (path !== '/chat' && path !== '/public-chat') {
    send(res, 404, { error: 'not_found', message: `No route for ${path}` });
    return;
  }

  if (req.method !== 'POST') {
    send(res, 405, { error: 'method_not_allowed', message: 'Use POST.' });
    return;
  }

  let status = 500;
  try {
    const body = await readBody(req);
    let upstreamRequest;

    if (path === '/public-chat') {
      upstreamRequest = {
        host: UPSTREAM_HOST,
        path,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      };
    } else if (AUTH_MODE === 'cognito') {
      const authorization = req.headers.authorization;
      if (typeof authorization !== 'string' || !authorization.startsWith('Bearer ')) {
        status = 401;
        send(res, status, { error: 'authentication_required', message: 'Sign in before chatting.' });
        return;
      }
      upstreamRequest = {
        host: UPSTREAM_HOST,
        path,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: authorization,
        },
      };
    } else {
      const creds = await credentials();
      upstreamRequest = aws4.sign(
        {
          host: UPSTREAM_HOST,
          path: '/chat',
          method: 'POST',
          service: SERVICE,
          region: REGION,
          body,
          headers: { 'Content-Type': 'application/json' },
        },
        {
          accessKeyId: creds.accessKeyId,
          secretAccessKey: creds.secretAccessKey,
          sessionToken: creds.sessionToken,
        },
      );
    }

    status = await forward(res, upstreamRequest, body);
  } catch (error) {
    cached = null;
    const message = error instanceof Error ? error.message : String(error);
    const isCredentials = /credential|profile|token|expired|sso/i.test(message);
    status = isCredentials ? 401 : 500;
    send(res, status, {
      error: isCredentials ? 'credentials_unavailable' : 'proxy_error',
      message: isCredentials
        ? `No usable AWS credentials (${message}). Try: aws sso login --profile "\${AWS_PROFILE:-default}"`
        : message,
    });
  } finally {
    console.log(`[proxy] ${req.method} ${path} -> ${status} (${Date.now() - started}ms)`);
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(
    AUTH_MODE === 'cognito'
      ? `[proxy] forwarding Cognito bearer tokens to ${UPSTREAM_HOST}`
      : `[proxy] signing ${SERVICE}/${REGION} requests for ${UPSTREAM_HOST}`,
  );
  console.log(`[proxy] listening on http://127.0.0.1:${PORT}`);
});
