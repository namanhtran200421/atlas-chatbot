import { TestBed } from '@angular/core/testing';
import { DomSanitizer } from '@angular/platform-browser';

import { MarkdownPipe } from './markdown.pipe';

describe('MarkdownPipe', () => {
  let pipe: MarkdownPipe;
  let sanitizer: DomSanitizer;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [MarkdownPipe] });
    pipe = TestBed.inject(MarkdownPipe);
    sanitizer = TestBed.inject(DomSanitizer);
  });

  const html = (value: string) =>
    sanitizer.sanitize(1 /* SecurityContext.HTML */, pipe.transform(value)) ?? '';

  it('renders bold, links and inline code', () => {
    expect(html('**bold**')).toContain('<strong>bold</strong>');
    expect(html('[docs](https://example.com)')).toContain(
      '<a href="https://example.com" target="_blank" rel="noopener noreferrer">docs</a>',
    );
    expect(html('use `npm start`')).toContain('<code>npm start</code>');
  });

  it('renders both kinds of list', () => {
    expect(html('- one\n- two')).toBe('<ul><li>one</li><li>two</li></ul>');
    expect(html('1. one\n2. two')).toBe('<ol><li>one</li><li>two</li></ol>');
  });

  it('keeps paragraphs and headings apart', () => {
    expect(html('## Programs\n\nLine one\nLine two')).toBe(
      '<h4>Programs</h4><p>Line one<br>Line two</p>',
    );
  });

  it('escapes raw HTML in the model output', () => {
    const out = html('<img src=x onerror="alert(1)"> **safe**');
    expect(out).not.toContain('<img');
    expect(out).toContain('&lt;img');
    expect(out).toContain('<strong>safe</strong>');
  });

  it('does not turn a javascript: link into an anchor', () => {
    expect(html('[x](javascript:alert(1))')).not.toContain('<a ');
  });
});
