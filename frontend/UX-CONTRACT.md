# Oriana chat contract

- The screen is a single conversation view. The browser keeps the visible transcript, and it is the only conversation memory: each request carries the last four completed exchanges so follow-ups are understood, and a reload or New chat forgets them. Nothing is persisted.
- `ChatService` owns pending, success, error and retry state. Only one request may be in flight. A failed turn keeps its question for retry.
- The composer uses a native labeled textarea. Enter sends on desktop; Shift+Enter or a touch keyboard inserts a line. The send button is disabled while empty or busy.
- Suggested questions are buttons that submit their text. New chat clears the local transcript and is disabled during a request.
- Citations open their source in a new tab. Model output is escaped before the small Markdown subset is rendered.
- The login route uses Cognito email/password authentication. Its access token lives in `sessionStorage`, chat is route-protected, and sign out clears the browser session.
- `npm start` owns the development proxy and Angular server, reusing an existing Oriana page if present. The proxy signs with IAM by default or forwards Cognito bearer tokens when `CHAT_AUTH_MODE=cognito`; client code never holds AWS credentials.
- The conversation panel owns scrolling; the composer stays reachable. Errors are shown inline with a retry button. Focus is visible on all controls.
