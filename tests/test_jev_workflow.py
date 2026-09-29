"""Offline checks for JEV decisions and the graph branches that use them."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from typesafe_sdk import Choice, Noul, Score

from agents import sql_analyst as analyst
from utils.database import DatabaseUtils
from utils.jev_decision import RequestDecision, classify_request, score_table_relevance

CATALOG = [
    {"name": "public.orders", "description": "public.orders: id (integer), customer_id (integer)"},
    {"name": "public.customers", "description": "public.customers: id (integer), name (text)"},
]


class JEVDecisionTests(unittest.TestCase):
    @patch("utils.jev_decision.TypeSafeClient")
    def test_choice_and_score_share_one_jev_call(self, client_class):
        client = client_class.return_value.__enter__.return_value
        client.system_one.return_value = SimpleNamespace(
            choices={"route": SimpleNamespace(choice="sql_question", confidence=0.91)},
            scores={"complexity": SimpleNamespace(score=1.8)},
        )

        decision = classify_request("How many orders?", CATALOG)

        self.assertEqual(decision, RequestDecision("sql_question", 0.91, 1.8))
        call = client.system_one.call_args.kwargs
        self.assertEqual(call["state"]["available_tables"], CATALOG)
        self.assertIsInstance(call["questions"]["route"], Choice)
        self.assertIsInstance(call["questions"]["complexity"], Score)

    @patch("utils.jev_decision.TypeSafeClient")
    def test_noul_scores_each_table(self, client_class):
        client = client_class.return_value.__enter__.return_value
        client.system_one.return_value = SimpleNamespace(
            nouls={
                "table_0": SimpleNamespace(noul=0.8),
                "table_1": SimpleNamespace(noul=0.3),
            }
        )

        result = score_table_relevance("How many orders?", CATALOG)

        self.assertEqual(result, {"public.orders": 0.8, "public.customers": 0.3})
        client.system_one.assert_called_once()
        self.assertIsInstance(client.system_one.call_args.kwargs["questions"]["table_0"], Noul)


class CatalogTests(unittest.TestCase):
    @patch("utils.database.psycopg2.connect")
    def test_catalog_contains_column_metadata_without_sample_rows(self, connect):
        cursor = connect.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            ("orders", "id", "integer"),
            ("orders", "customer_id", "integer"),
        ]

        catalog = DatabaseUtils({}).table_catalog("analytics")

        self.assertEqual(
            catalog,
            [
                {
                    "name": "analytics.orders",
                    "description": "analytics.orders: id (integer), customer_id (integer)",
                }
            ],
        )
        self.assertEqual(cursor.execute.call_args.args[1], ("analytics",))
        self.assertNotIn("SELECT *", cursor.execute.call_args.args[0])


class GraphRoutingTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "host": "localhost",
                "port": "5432",
                "user": "test",
                "password": "test",
                "database": "test",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

        database_patch = patch.object(analyst, "DatabaseUtils")
        database_class = database_patch.start()
        self.addCleanup(database_patch.stop)
        database_class.return_value.table_catalog.return_value = CATALOG
        database_class.return_value.schema_details.return_value = "Table: orders; Table: customers"
        database_class.return_value.execute_sql.return_value = "[(4,)]"
        self.database_class = database_class

    def invoke(self, question):
        return analyst.sql_analyst.invoke({"user_question": question})

    @patch.object(analyst, "classify_request")
    def test_clarification_skips_sql(self, classify):
        classify.return_value = RequestDecision("clarification", 0.85, 0.0)
        with patch.object(analyst, "pick_llm") as llm:
            result = self.invoke("How many?")

        self.assertIn("clarify", result["final_answer"].lower())
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["request_route"], "clarification")
        llm.assert_not_called()

    @patch.object(analyst, "classify_request")
    def test_out_of_scope_uses_current_table_names(self, classify):
        classify.return_value = RequestDecision("out_of_scope", 0.9, 0.0)
        result = self.invoke("Write a poem")

        self.assertIn("public.orders", result["final_answer"])
        self.assertIn("public.customers", result["final_answer"])

    @patch.object(analyst, "classify_request")
    def test_empty_catalog_stops_before_jev(self, classify):
        self.database_class.return_value.table_catalog.return_value = []

        result = self.invoke("How many orders?")

        self.assertIn("no tables", result["final_answer"])
        classify.assert_not_called()

    @patch.object(analyst, "score_table_relevance")
    @patch.object(analyst, "classify_request")
    def test_sql_path_uses_jev_hints_and_keeps_one_answer(self, classify, relevance):
        classify.return_value = RequestDecision("sql_question", 0.9, 1.8)
        relevance.return_value = {"public.orders": 0.8, "public.customers": 0.3}
        llm = MagicMock()
        llm.invoke.side_effect = [
            SimpleNamespace(content="How many orders are there?"),
            SimpleNamespace(content="SELECT count(*) FROM public.orders"),
            SimpleNamespace(content="There are 4 orders."),
        ]
        llm.with_structured_output.return_value.invoke.return_value.model_dump.return_value = {
            "answer": "Yes",
            "comments": "Read-only query",
        }
        with patch.object(analyst, "pick_llm", return_value=llm):
            result = self.invoke("How many orders?")

        self.assertEqual(result["selected_tables"], ["public.orders", "public.customers"])
        self.assertIn(
            "JEV-suggested tables: public.orders, public.customers", result["prompt_query_context"]
        )
        self.assertIn("multiple joins", result["prompt_query_context"])
        self.assertEqual(result["final_answer"], "There are 4 orders.")
        self.assertEqual(len(result["messages"]), 1)
        self.database_class.return_value.execute_sql.assert_called_once()


if __name__ == "__main__":
    unittest.main()
