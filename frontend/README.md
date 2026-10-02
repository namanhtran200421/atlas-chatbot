# Oriana — Membership Atlas chat (Angular)

Frontend for the Membership Atlas RAG pipeline in [`../membership_atlas`](../membership_atlas).
It talks to the `membership-rag-api` Lambda (`membership_rag.lambda_handler.handler`)
behind the `membership-rag-http-api` HTTP API in `ap-southeast-2`.

## Run it

```bash
npm install
npm start          # starts/reuses the signing proxy and Angular → http://localhost:4200
npm run build      # production bundle into dist/
npm test           # unit tests (vitest)
```

## Cognito configuration

The `/login` route uses Cognito's `USER_PASSWORD_AUTH` flow. Configure the
public app-client ID in both environment files before signing in:

```ts
// src/environments/environment.development.ts (and environment.ts for production)
cognitoRegion: 'ap-southeast-2',
cognitoUserPoolId: 'ap-southeast-2_79ScAKqng',
cognitoClientId: '2aj4cq4fv29f73m28ttc3c9n27',
```

The Cognito app client must have `ALLOW_USER_PASSWORD_AUTH` enabled and must
not have a client secret. The browser stores the access token in
`sessionStorage`, protects the chat route, and sends
`Authorization: Bearer <access-token>` with chat requests. Sign out removes the
browser session; access tokens are not persisted across tabs.

The `/signup` route creates a Cognito user with the email address as the
username, then asks for the confirmation code Cognito sends by email. It also
supports resending the code. For this to work, configure the user pool in the
AWS console under **Sign-up**:

1. Enable **Self-service sign-up**.
2. Enable Cognito-assisted verification and confirmation for **email**.
3. Confirm that the public app client has no client secret.

Self-service registration is public: anyone who can reach the app and knows the
app-client ID can create a user in the pool. Review that access model before
deploying it beyond testing.

The currently documented Membership API route uses `AWS_IAM`. In that mode,
`npm start` continues to sign upstream requests with your local AWS credentials;
the Cognito login is a client-side gate until the API Gateway route is changed
to use the Cognito authorizer. After that change, start the proxy in bearer-token
forwarding mode:

```powershell
$env:CHAT_AUTH_MODE = 'cognito'
npm start
```

API Gateway must validate tokens from the same user pool and app client. If it
is a REST API Cognito authorizer, confirm whether the method expects an access
token with OAuth scopes or an ID token before deployment; this frontend ports
the branch's existing access-token behaviour.

`npm start` starts the signer on port 4300 and Angular on port 4200. It also
works if an Oriana Angular page is already running: it starts whichever service
is missing. Opening the page with `ng serve` alone leaves the signer offline and
chat requests fail with 502. You need working AWS credentials (`aws login` or
`aws sso login` for your profile), or the signer returns
`401 credentials_unavailable` and the UI explains the next step.

## Why there is a proxy

`POST /chat` is currently protected with **AWS_IAM**, so every request must be
SigV4-signed. A browser cannot safely hold those credentials, so the default
development mode signs on your machine:

```
browser ──POST /api/chat──▶ ng serve proxy ──▶ tools/aws-proxy.mjs
                                                      │ SigV4 (your AWS creds)
                                                      ▼
                          may54zk1a5.execute-api.ap-southeast-2.amazonaws.com/chat
```

The app only ever calls the same-origin `/api/chat`, so there is no CORS in play
at all. `aws4` and `@aws-sdk/credential-providers` are **devDependencies** —
nothing about signing ships to the browser.

For Cognito-authorized development, the same proxy forwards the browser's bearer
token when `CHAT_AUTH_MODE=cognito`. Production can call a Cognito-protected API
Gateway endpoint directly when CORS allows the deployed origin, or use a
same-origin reverse proxy. The alternative is making the route public — but the handler grants every
caller all three access classes (`public`, `member_restricted`, `configuration`),
so that would expose the whole corpus.

## Backend contract

The endpoint **stores nothing**, so the memory is the page: every request
replays the last completed turns (up to fifty exchanges) from the on-screen transcript
as `history`, and the Lambda uses them to resolve what a follow-up refers to. It
lives in a signal only — a reload, a new tab or **New chat** starts Oriana over,
and nothing is written to `localStorage`. The first question of a session sends
no `history` field at all.

```jsonc
// request
{
  "query": "How much is it?",
  "number_of_results": 5,
  "history": [
    { "role": "user", "content": "What membership plans are available?" },
    { "role": "assistant", "content": "There are four plans." }
  ]
}

// 200
{
  "answer": "We offer four membership options…",
  "citations": [{ "title": "Membership Levels", "url": "https://…" }],
  "request_id": "c134280b-…"
}

// error (4xx/5xx)
{ "error": "request_not_allowed", "message": "…", "request_id": "…" }
```

Two server-side limits the client respects:

| Limit               | Value      | Set by                                           |
| ------------------- | ---------- | ------------------------------------------------ |
| `number_of_results` | 1–5        | `MEMBERSHIP_RAG_MAX_RESULTS` on the Lambda       |
| query length        | 4000 chars | `membership_rag.guardrails.MAX_QUERY_CHARACTERS` |

Guardrail refusals and out-of-scope questions come back as a normal `200` with
Oriana declining in her own words — only malformed or blocked requests are
non-2xx, and each error code maps to a plain-English line in
[`chat.service.ts`](src/app/chat/chat.service.ts).

## Layout

| Path                                                             | Purpose                                                         |
| ---------------------------------------------------------------- | --------------------------------------------------------------- |
| [`tools/dev.mjs`](tools/dev.mjs)                                 | Starts or reuses the local page and signer                      |
| [`tools/aws-proxy.mjs`](tools/aws-proxy.mjs)                     | Dev-only SigV4 signing proxy                                    |
| [`proxy.conf.json`](proxy.conf.json)                             | Routes `/api` from `ng serve` to the proxy                      |
| [`src/app/chat/chat.ts`](src/app/chat/chat.ts)                   | Chat screen — composer, scrolling, copy/retry                   |
| [`src/app/chat/chat.service.ts`](src/app/chat/chat.service.ts)   | Transcript, HTTP calls, error mapping                           |
| [`src/app/chat/markdown.pipe.ts`](src/app/chat/markdown.pipe.ts) | Escapes model output, then renders the Markdown subset it emits |
| [`src/app/signup/signup.ts`](src/app/signup/signup.ts)           | Self-service Cognito sign-up and email-code confirmation        |
| [`src/styles.scss`](src/styles.scss)                             | Atlas-inspired color/type tokens and Markdown prose styles      |

The app is zoneless (no `zone.js`); all state is held in signals.

## Notes

- Model output is HTML-escaped before any Markdown is applied, and only `http(s)`
  links become anchors — see [`markdown.pipe.spec.ts`](src/app/chat/markdown.pipe.spec.ts).
- The visual design follows the Cultural Infusion Atlas site: white canvas, mixed Montserrat and italic Piazzolla display type, lavender panels and rounded buttons. The local copy of the Atlas logo is in `public/`.
- To point at another deployment, set `MEMBERSHIP_RAG_HOST` (and `AWS_REGION`)
  before `npm run proxy`.
