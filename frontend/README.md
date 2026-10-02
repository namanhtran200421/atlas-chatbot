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
`sessionStorage` and sends `Authorization: Bearer <access-token>` with
signed-in chat requests. Guests can use public chat without signing in. Sign
out removes the browser session; access tokens are not persisted across tabs.

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

API Gateway validates Cognito access tokens on `POST /chat`. New accounts can
search public content. Membership in the Cognito `members` group grants access
to member-restricted content. Guests use `POST /public-chat` for public content.

`npm start` starts the local API proxy on port 4300 and Angular on port 4200.
Opening the page with `ng serve` alone leaves the proxy offline. The proxy
forwards bearer tokens for signed-in chat and sends public chat without a token.

## Why there is a proxy

The frontend uses same-origin API paths in development and production:

```
browser ──POST /api/chat────────▶ local proxy or Vercel rewrite ──▶ API Gateway /chat
        └─POST /api/public-chat─▶ local proxy or Vercel rewrite ──▶ API Gateway /public-chat
```

Vercel's production rewrites are in `vercel.json`. They forward the
Authorization header for the protected route. API Gateway also allows the
production Vercel origin for direct browser API calls.

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
