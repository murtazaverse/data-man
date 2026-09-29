"""Streamlit interface for the existing SQL analyst graph."""

import logging
import os

import streamlit as st

from agents.sql_analyst import sql_analyst

logger = logging.getLogger(__name__)


def show_message(message: dict[str, str]) -> None:
    """Render one saved turn without rerunning the analyst."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sql"):
            with st.expander("SQL and database result"):
                st.code(message["sql"], language="sql")
                if message.get("result"):
                    st.code(message["result"], language="text")


def main() -> None:
    st.set_page_config(page_title="Data-Man", page_icon="🕷️")
    st.title("🕷️ Data-Man")
    st.caption("Ask a question about your PostgreSQL data in plain English.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    with st.sidebar:
        st.header("Settings")
        schema = st.text_input("Database schema", value="public").strip() or "public"
        st.caption("The schema applies to new questions. Each question is answered independently.")
        if st.button("Clear chat"):
            st.session_state.conversation = []
            st.rerun()

        missing = [
            name for name in ("host", "port", "user", "password", "database") if not os.getenv(name)
        ]
        if missing:
            st.warning(f"Missing database settings in .env: {', '.join(missing)}")

    for message in st.session_state.conversation:
        show_message(message)

    question = st.chat_input("Ask about your data")
    if not question or not question.strip():
        return

    question = question.strip()
    user_message = {"role": "user", "content": question}
    st.session_state.conversation.append(user_message)
    show_message(user_message)

    with st.spinner("Analyzing your question..."):
        try:
            response = sql_analyst.invoke({"user_question": question, "database_schema": schema})
            assistant_message = {
                "role": "assistant",
                "content": response.get("final_answer") or "The analyst did not return an answer.",
                "sql": response.get("generated_sql_query") or "",
                "result": response.get("sql_query_execution_result") or "",
            }
        except Exception:
            logger.exception("SQL analyst request failed")
            assistant_message = {
                "role": "assistant",
                "content": (
                    "I could not complete that request. Check the database connection and "
                    "model/JEV credentials, then try again."
                ),
            }

    st.session_state.conversation.append(assistant_message)
    show_message(assistant_message)


if __name__ == "__main__":
    main()
