/**
 * Wire types for the Membership Atlas RAG Lambda
 * (membership_rag.lambda_handler.handler behind POST /chat).
 *
 * The endpoint stores nothing. Conversation memory lives in this page: each
 * request carries the recent turns the tab still holds, so a reload starts a
 * new conversation.
 */

/** One earlier message, replayed so a follow-up can be understood. */
export interface HistoryTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  query: string;
  number_of_results: number;
  history?: HistoryTurn[];
  conversation_state?: string;
}

/** A public source backing an answer. */
export interface Citation {
  title: string;
  url: string;
}

export interface ChatResponse {
  answer: string;
  citations?: Citation[];
  request_id?: string;
  conversation_state?: string;
}

/** Error body shared by the Lambda and the local signing proxy. */
export interface ChatErrorResponse {
  error: string;
  message?: string;
  request_id?: string;
}

export type MessageStatus = 'pending' | 'done' | 'error';

/** A message as the UI holds it. */
export interface UiMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  status: MessageStatus;
  at: number;
  /** The static welcome message has no copy action. */
  isIntro?: boolean;
  citations?: Citation[];
  /** Correlates a turn with the Lambda's CloudWatch log line. */
  requestId?: string;
  /** On a failed turn, the question to resend when retrying. */
  retryText?: string;
}
