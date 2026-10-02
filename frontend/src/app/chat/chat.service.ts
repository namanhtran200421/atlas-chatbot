import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { firstValueFrom, timeout } from 'rxjs';

import { environment } from '../../environments/environment';
import { AuthService } from '../auth/auth.service';
import {
  ChatErrorResponse,
  ChatRequest,
  ChatResponse,
  HistoryTurn,
  UiMessage,
} from './chat.models';

const GREETING = "Hey, I'm **Oriana**! Ask me about membership, culture or events.";

/** Friendly text per error code returned by the Lambda or the signing proxy. */
const ERROR_TEXT: Record<string, string> = {
  request_not_allowed:
    "I can't help with that one — ask me something about Membership Atlas or our cultural events instead.",
  invalid_query: 'That question came through empty. Try typing it again?',
  invalid_number_of_results: 'That request asked for more sources than the service allows.',
  service_unavailable:
    'The Membership Atlas service is not configured right now. Please try again later.',
  answer_unavailable:
    'The answer service is temporarily unavailable. Please try again in a moment.',
  retrieval_safety_error:
    "I couldn't safely process that result. Please try rephrasing your question.",
  credentials_unavailable:
    'The local signing proxy has no usable AWS credentials. Run `aws sso login` and restart `npm start`.',
  upstream_unreachable:
    "I couldn't reach the Membership Atlas API. Check your connection and try again.",
};

/** Completed turns replayed for context — fifty exchanges, newest last. */
const HISTORY_TURNS = 100;

let messageCounter = 0;
const nextId = () => `m${++messageCounter}`;

/**
 * Owns the on-screen transcript, which doubles as the conversation memory.
 *
 * The backend stores nothing, so each request replays the last few completed
 * turns as `history`. That memory lives only in this signal: a reload, a new
 * tab or "New chat" starts the conversation over, and nothing is written to
 * storage. The Lambda caps and screens whatever is sent.
 */
@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly http = inject(HttpClient);
  private readonly auth = inject(AuthService);

  private readonly _messages = signal<UiMessage[]>([this.greeting()]);
  private readonly _isSending = signal(false);
  private conversationState: string | null = null;
  private accessMode: 'public' | 'member' = this.auth.accessToken() ? 'member' : 'public';

  readonly messages = this._messages.asReadonly();
  readonly isSending = this._isSending.asReadonly();
  /** True until the first question — drives the welcome screen. */
  readonly isEmpty = computed(() => this._messages().every((m) => m.role === 'assistant'));

  private greeting(): UiMessage {
    return { id: nextId(), role: 'assistant', content: GREETING, status: 'done', at: Date.now(), isIntro: true };
  }

  reset(): void {
    if (this._isSending()) return;
    messageCounter = 0;
    this.conversationState = null;
    this.accessMode = this.auth.accessToken() ? 'member' : 'public';
    this._messages.set([this.greeting()]);
  }

  async send(text: string): Promise<void> {
    const query = text.trim();
    if (!query || this._isSending()) return;

    const accessMode = this.auth.accessToken() ? 'member' : 'public';
    if (accessMode !== this.accessMode) this.reset();
    const url = accessMode === 'member' ? environment.chatApiUrl : environment.publicChatApiUrl;

    this._isSending.set(true);
    const placeholderId = nextId();
    // Captured before the new turn is appended, so it holds only what was
    // already said and answered.
    const history = this.history();

    this._messages.update((list) => [
      ...list,
      { id: nextId(), role: 'user', content: query, status: 'done', at: Date.now() },
      { id: placeholderId, role: 'assistant', content: '', status: 'pending', at: Date.now() },
    ]);

    try {
      const body: ChatRequest = {
        query,
        number_of_results: environment.numberOfResults,
        ...(this.conversationState
          ? { conversation_state: this.conversationState }
          : history.length ? { history } : {}),
      };

      const data = await firstValueFrom(
        this.http
          .post<ChatResponse>(url, body)
          .pipe(timeout(environment.requestTimeoutMs)),
      );

      this.conversationState = data?.conversation_state ?? null;

      this.patch(placeholderId, {
        content: (data?.answer ?? '').trim() || "I couldn't put an answer together for that one.",
        status: 'done',
        citations: data?.citations?.filter((c) => c?.url) ?? [],
        requestId: data?.request_id,
      });
    } catch (error) {
      this.patch(placeholderId, {
        content: this.describe(error),
        status: 'error',
        requestId: this.requestIdOf(error),
        retryText: query,
      });
    } finally {
      this._isSending.set(false);
    }
  }

  /** Drops the failed exchange and asks the same question again. */
  async retry(failedId: string): Promise<void> {
    const failed = this._messages().find((m) => m.id === failedId);
    if (!failed?.retryText || this._isSending()) return;

    const index = this._messages().findIndex((m) => m.id === failedId);
    // Remove the error bubble and the question that produced it.
    this._messages.update((list) => list.filter((_, i) => i !== index && i !== index - 1));
    await this.send(failed.retryText);
  }

  /**
   * The turns worth replaying: the greeting is not part of the conversation,
   * and a pending or failed bubble has no content to carry.
   */
  private history(): HistoryTurn[] {
    return this._messages()
      .filter((m) => !m.isIntro && m.status === 'done' && m.content.trim())
      .slice(-HISTORY_TURNS)
      .map((m) => ({ role: m.role, content: m.content }));
  }

  private patch(id: string, patch: Partial<UiMessage>): void {
    this._messages.update((list) =>
      list.map((m) => (m.id === id ? { ...m, ...patch, at: Date.now() } : m)),
    );
  }

  private requestIdOf(error: unknown): string | undefined {
    return error instanceof HttpErrorResponse
      ? (error.error as ChatErrorResponse | null)?.request_id
      : undefined;
  }

  private describe(error: unknown): string {
    if (error instanceof HttpErrorResponse) {
      if (error.status === 0) {
        return "I couldn't reach the Membership Atlas API. Check your connection and try again.";
      }
      if (error.status === 403 && !error.error?.error) {
        return 'Your sign-in is not authorised for this request. Please sign in again.';
      }
      const code = (error.error as ChatErrorResponse | null)?.error ?? '';
      if (error.status === 502 && !code) {
        return 'The chat service is temporarily unavailable. Please try again.';
      }
      return ERROR_TEXT[code] ?? `Something went wrong (${error.status}). Please try again.`;
    }
    if (error instanceof Error && error.name === 'TimeoutError') {
      return 'That took too long to answer. Try again, or narrow the question a little.';
    }
    return 'Something went wrong on the way to Membership Atlas. Please try again.';
  }
}
