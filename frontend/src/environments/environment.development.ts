export const environment = {
  production: false,
  /** Public Cognito app-client configuration; never put a client secret here. */
  cognitoRegion: 'ap-southeast-2',
  cognitoUserPoolId: 'ap-southeast-2_79ScAKqng',
  cognitoClientId: '2aj4cq4fv29f73m28ttc3c9n27',
  /** Served by tools/aws-proxy.mjs via proxy.conf.json — see README.md. */
  chatApiUrl: '/api/chat',
  publicChatApiUrl: '/api/public-chat',
  numberOfResults: 5,
  maxQueryLength: 4000,
  requestTimeoutMs: 45_000,
};
