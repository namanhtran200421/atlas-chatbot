"""Run the Membership RAG CLI without installing the console-script entry point."""

from membership_rag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
