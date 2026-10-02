/**
 * Default (production) environment.
 * Replaced by environment.development.ts for `ng serve` / `ng build --configuration development`.
 */
export const environment = {
  production: true,

  /** Cognito app clients used in a browser must not have a client secret. */
  cognitoRegion: 'ap-southeast-2',
  cognitoUserPoolId: 'ap-southeast-2_79ScAKqng',
  cognitoClientId: '2aj4cq4fv29f73m28ttc3c9n27',

  /**
   * Membership Atlas RAG endpoint, relative on purpose.
   *
   * The upstream API is https://may54zk1a5.execute-api.ap-southeast-2.amazonaws.com/chat,
   * whose POST /chat route is protected with AWS_IAM — every call must be
   * SigV4-signed, which a browser cannot do on its own. So the app always talks
   * to a same-origin /api/chat and whatever serves it does the signing:
   * tools/aws-proxy.mjs in development, and a Cognito identity pool or a
   * server-side signer in production. See README.md.
   */
  chatApiUrl: '/api/chat',

  /** Retrieval breadth. The Lambda caps this at MEMBERSHIP_RAG_MAX_RESULTS (currently 5). */
  numberOfResults: 5,

  /** Guardrail limit in membership_rag.guardrails.MAX_QUERY_CHARACTERS. */
  maxQueryLength: 4000,

  /** Abandon a turn after this long; the Lambda itself times out at 30s. */
  requestTimeoutMs: 45_000,
};
