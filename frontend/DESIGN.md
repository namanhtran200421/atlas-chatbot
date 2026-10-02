---
version: alpha
name: Oriana Membership Atlas
description: Atlas-inspired chat for Membership Atlas
colors:
  primary: '#543EB6'
  background: '#FFFFFF'
  surface: '#F0EDFF'
  ink: '#1C1C1C'
  muted: '#55535C'
  danger: '#AE322A'
typography:
  display:
    fontFamily: 'Piazzolla, Georgia, serif'
  body:
    fontFamily: 'Montserrat, Arial, sans-serif'
rounded:
  DEFAULT: '18px'
spacing:
  page-gutter: 'clamp(22px, 5vw, 64px)'
components:
  prompt-card: {}
  message: {}
  composer: {}
---

# Oriana Design System

## Direction

Oriana visually belongs to [Cultural Infusion Atlas](https://atlas.culturalinfusion.com/). Use the official Atlas logo, a spacious white canvas, restrained lavender surfaces, black and outlined pill controls, and the site's mixed display language: a light sans heading with bold italic serif emphasis. The chat stays quiet and readable as answers grow. The display treatment is for short headings only, not answer text.

## Colors and type

Ink is `#1C1C1C`; lavender `#F0EDFF` frames the question area; purple `#543EB6` marks links and small labels. Montserrat carries interface and answer text. Italic Piazzolla carries emphasized headline words. Error states use `#AE322A` plus a retry action. Runtime values live in `src/styles.scss`.

## Layout and interaction

The narrow top bar carries the Atlas logo and a fresh-chat action. The empty state has a centered introduction above a lavender question panel. After a question, a smaller heading leaves room for the transcript. The transcript alone scrolls; the composer stays at the bottom. Suggested questions stack on phones. User messages are dark, assistant messages are white, both with visible speaker labels. Sources are links. Keyboard focus is purple and visible, and reduced motion preference removes animation.
