"""Small, database-independent decisions made with TypeSafe AI's JEV model."""

from dataclasses import dataclass
from typing import Literal

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

Route = Literal["sql_question", "clarification", "out_of_scope"]


@dataclass(frozen=True)
class RequestDecision:
    """The two decisions JEV makes about one incoming question."""

    route: Route
    confidence: float
    complexity_score: float


def classify_request(question: str, catalog: list[dict[str, str]]) -> RequestDecision:
    """Route a question and rate its complexity against the connected database."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    if not catalog:
        raise ValueError("A database table catalog is required to classify the question.")

    with TypeSafeClient() as client:
        response = client.system_one(
            state={"question": question, "available_tables": catalog},
            questions={
                "route": Choice(
                    instructions=(
                        "How should a read-only data assistant handle this question, "
                        "given only the available database tables?"
                    ),
                    criteria={
                        "sql_question": (
                            "A specific question likely answerable by querying the available data."
                        ),
                        "clarification": (
                            "A question about the available data that needs a missing detail "
                            "or has an unclear meaning."
                        ),
                        "out_of_scope": (
                            "A request unrelated to the available data or asking the assistant "
                            "to do something other than read and explain it."
                        ),
                    },
                ),
                "complexity": Score(
                    instructions="How much analysis would answering this question require?",
                    criteria=[
                        "A straightforward lookup or count",
                        "An aggregation, grouping, or simple comparison",
                        "Several joins, comparisons, or analysis steps",
                    ],
                ),
            },
        )

    route = response.choices["route"]
    return RequestDecision(
        route=route.choice,
        confidence=route.confidence,
        complexity_score=response.scores["complexity"].score,
    )


def score_table_relevance(question: str, catalog: list[dict[str, str]]) -> dict[str, float]:
    """Ask whether each table may help answer the question, including through joins.

    Later, a RAG retriever can pass its small candidate list to this same function.
    """
    if len(catalog) == 1:
        return {catalog[0]["name"]: 1.0}

    questions = {
        f"table_{index}": Noul(
            instructions=(
                f"Could {table['description']} help answer the question, "
                "either directly or through a join to another table?"
            )
        )
        for index, table in enumerate(catalog)
    }
    with TypeSafeClient() as client:
        response = client.system_one(state={"question": question}, questions=questions)
    return {
        table["name"]: response.nouls[f"table_{index}"].noul for index, table in enumerate(catalog)
    }
