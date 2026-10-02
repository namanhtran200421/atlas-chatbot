import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  effect,
  inject,
  signal,
  viewChild,
} from '@angular/core';
import { Router } from '@angular/router';

import { environment } from '../../environments/environment';
import { AuthService } from '../auth/auth.service';
import { ChatService } from './chat.service';
import { MarkdownPipe } from './markdown.pipe';

const SUGGESTIONS = [
  'What membership plans are available?',
  'What is National Reconciliation Week?',
  'How does Atlas describe being an effective ally?',
  'What types of neurodivergence does Atlas cover?',
];

@Component({
  selector: 'app-chat',
  imports: [MarkdownPipe],
  templateUrl: './chat.html',
  styleUrl: './chat.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Chat {
  protected readonly chat = inject(ChatService);
  protected readonly auth = inject(AuthService);
  protected readonly suggestions = SUGGESTIONS;
  protected readonly maxQueryLength = environment.maxQueryLength;

  protected readonly draft = signal('');
  protected readonly copiedId = signal<string | null>(null);
  /** Hides the "jump to latest" affordance while the newest turn is in view. */
  protected readonly atBottom = signal(true);

  private readonly scroller = viewChild.required<ElementRef<HTMLElement>>('scroller');
  private readonly input = viewChild.required<ElementRef<HTMLTextAreaElement>>('input');
  private readonly router = inject(Router);

  constructor() {
    // Follow the conversation as it grows, unless the reader has scrolled away.
    effect(() => {
      this.chat.messages();
      if (!this.chat.isEmpty() && this.atBottom()) this.scrollToLatest();
    });
  }

  protected onInput(event: Event): void {
    const el = event.target as HTMLTextAreaElement;
    this.draft.set(el.value);
    this.autoGrow(el);
  }

  protected onKeydown(event: KeyboardEvent): void {
    // Enter sends; Shift+Enter (and every Enter on a soft keyboard) adds a line.
    if (event.key === 'Enter' && !event.shiftKey && !this.isTouch()) {
      event.preventDefault();
      void this.send();
    }
  }

  protected onScroll(): void {
    const el = this.scroller().nativeElement;
    this.atBottom.set(el.scrollHeight - el.scrollTop - el.clientHeight < 80);
  }

  protected async send(): Promise<void> {
    const text = this.draft().trim();
    if (!text || this.chat.isSending()) return;

    this.draft.set('');
    const el = this.input().nativeElement;
    el.value = '';
    this.autoGrow(el);
    this.atBottom.set(true);

    await this.chat.send(text);
    if (!this.isTouch()) el.focus();
  }

  protected ask(prompt: string): void {
    this.draft.set(prompt);
    void this.send();
  }

  protected retry(id: string): void {
    void this.chat.retry(id);
  }

  protected newChat(): void {
    this.chat.reset();
    this.draft.set('');
    const el = this.input().nativeElement;
    el.value = '';
    this.autoGrow(el);
  }

  protected signOut(): void {
    this.auth.signOut();
    void this.router.navigateByUrl('/login');
  }

  protected async copy(id: string, content: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(content);
      this.copiedId.set(id);
      setTimeout(() => this.copiedId.update((v) => (v === id ? null : v)), 1600);
    } catch {
      // Clipboard blocked (insecure context or denied) — nothing useful to show.
    }
  }

  protected scrollToLatest(): void {
    requestAnimationFrame(() => {
      const el = this.scroller().nativeElement;
      el.scrollTop = el.scrollHeight;
      this.atBottom.set(true);
    });
  }

  private autoGrow(el: HTMLTextAreaElement): void {
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 168)}px`;
  }

  private isTouch(): boolean {
    return matchMedia('(hover: none) and (pointer: coarse)').matches;
  }
}
