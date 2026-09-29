### 🕷️ Data-Man

**Your friendly neighbourhood Data Agent.**

Ask a question in plain English. Data-Man takes care of the rest.

## Run the Streamlit app

1. Install dependencies with `uv sync` (or `pip install -r requirements.txt`).
2. Add PostgreSQL settings (`host`, `port`, `user`, `password`, `database`),
   `GROQ_API_KEY`, and `TYPESAFE_API_KEY` to a `.env` file in the project root.
3. Run `uv run streamlit run streamlit_app.py` (or `streamlit run streamlit_app.py`
   if you installed with pip).

The sidebar lets you choose a PostgreSQL schema (default: `public`) and clear the
visible conversation. Each question runs independently through the existing SQL
analyst graph. The assistant answer appears in chat; when SQL was generated, you
can expand the message to inspect the SQL and raw database result.
