import pytest

from membership_rag.bedrock.generation import BedrockAnswerGenerator
from membership_rag.bedrock.retrieval import RetrievedChunk
from membership_rag.privacy import (
    contains_email_address,
    contains_personal_membership_claim,
    is_personal_account_page,
)


def test_personal_account_urls_are_excluded() -> None:
    assert is_personal_account_page("https://membership.example.com/membership-account/")
    assert is_personal_account_page("https://membership.example.com/login/")
    assert not is_personal_account_page("https://membership.example.com/privacy-policy/")


def test_personal_details_are_detected_in_answers() -> None:
    assert contains_email_address("Contact: example.person@example.com")
    assert contains_personal_membership_claim("Example Person is a member of Atlas.")
    assert not contains_personal_membership_claim("We offer membership plans for teams.")


@pytest.mark.parametrize(
    "answer",
    [
        # Describing the product is not a disclosure about a person. This
        # suppressed the answer to "what is Cultural Infusion?".
        "Cultural Infusion is a membership organisation supporting diversity.",
        "Membership Atlas is a membership platform for workforce insights.",
        "The Free Tour is a membership tier you can join.",
        "Diversity Atlas is a member of our product family.",
    ],
)
def test_describing_the_product_is_not_a_personal_claim(answer: str) -> None:
    assert not contains_personal_membership_claim(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "John Smith is a member of the organisation.",
        "Jane Doe has a membership account.",
        "Example Person is a member.",
    ],
)
def test_a_named_person_membership_claim_is_still_caught(answer: str) -> None:
    assert contains_personal_membership_claim(answer)


class FakeGeneratorClient:
    def converse(self, **kwargs):
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": (
                                "Example Person is a member with the email "
                                "example.person@example.com.\nSOURCE_IDS: none"
                            )
                        }
                    ]
                }
            }
        }


def test_generated_personal_details_are_replaced_before_response() -> None:
    generator = BedrockAnswerGenerator(
        region_name="ap-southeast-2", client=FakeGeneratorClient()
    )

    source = RetrievedChunk(
        text="Public membership plan details.",
        score=0.9,
        metadata={"title": "Membership plans", "source_type": "membership_plans"},
        document_id="plans:1",
        source_uri=None,
    )
    answer = generator.generate("What membership plans are available?", [source])

    assert "example.person@example.com" not in answer.text
    assert "Example Person" not in answer.text
    assert "can't verify or share" in answer.text
    assert answer.citations == ()
