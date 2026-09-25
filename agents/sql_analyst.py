from utils.llm_selection import pick_llm
from typing import Annotated, Literal
from models.schema import AgentSchema


# --------------- AI Code -----------

def curate_ques(state: AgentSchema)->AgentSchema:
    """
    *FIRST NODE*

    This just curates the user's question.
    Args: Taking in the State i.e. Agent Schema.
    Return: Also returns the state.

    Uses Low level model.
    """
    model = pick_llm("low")
    response = model.invoke(f"Curate the following question: {state.user_question}")
    state.curated_ques = response.content[0]['text']

    return state