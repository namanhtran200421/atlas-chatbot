from collections import defaultdict
from pathlib import Path

from membership_rag.corpus import (
    load_documents,
)

CORPUS_PATH = Path(
    "data/input/cleaned"
)

SAMPLES_PER_TYPE = 5


def main() -> None:
    documents = load_documents(
        CORPUS_PATH
    )

    grouped = defaultdict(list)

    for document in documents:
        grouped[
            document.content_type
        ].append(document)

    for content_type in sorted(
        grouped
    ):
        docs = grouped[
            content_type
        ]

        print()
        print("=" * 80)
        print(
            f"{content_type}: "
            f"{len(docs)} documents"
        )
        print("=" * 80)

        for document in docs[
            :SAMPLES_PER_TYPE
        ]:
            print()
            print(
                "DOCUMENT ID:",
                document.document_id,
            )

            print(
                "TITLE:",
                document.title,
            )

            print(
                "ACCESS:",
                document.access_class,
            )

            print(
                "SOURCE:",
                document.source_type,
            )

            print(
                "URL:",
                document.url,
            )

            preview = (
                document.content_markdown
                .replace("\n", " ")
                [:400]
            )

            print(
                "PREVIEW:",
                preview,
            )


if __name__ == "__main__":
    main()