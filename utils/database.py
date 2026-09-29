"""
Here we will create a class with which we can tell
how to connect with the server and how to fetch the data.

So the llm will get all the context from here.
Q: Why do we need to create all the context again and again?
A: So that the LLM could learn the current state of out data, if there are any data changes.
"""

import os

import psycopg2
from psycopg2 import sql


class DatabaseUtils:
    """
    Connection to the database.
    """

    def __init__(self, db_config):
        self.db_config = db_config

        try:
            self.connection = psycopg2.connect(**db_config)
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            self.connection = None

    def table_catalog(self, schema_name: str) -> list[dict[str, str]]:
        """Describe accessible tables using metadata only, without sample rows."""
        if self.connection is None:
            raise ConnectionError("The database connection is unavailable.")

        with self.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = %s
                ORDER BY table_name, ordinal_position
                """,
                (schema_name,),
            )
            rows = cursor.fetchall()

        columns_by_table: dict[str, list[str]] = {}
        for table_name, column_name, data_type in rows:
            columns_by_table.setdefault(table_name, []).append(f"{column_name} ({data_type})")

        return [
            {
                "name": f"{schema_name}.{table_name}",
                "description": f"{schema_name}.{table_name}: {', '.join(columns)}",
            }
            for table_name, columns in columns_by_table.items()
        ]

    def schema_details(self, schema_name):
        """
        Query to find the context from my database.
        I am preparing the context here for the LLM.
        (Use RAG here later)
        """
        schema_info_context = ""
        cursor = None
        conn = self.connection
        cursor = conn.cursor()  # The object that we get from the postgre connection, with the help of it we can run the queries.
        schema_info_context = (
            f"Database Schema: {schema_name}\n"  # We will keep on adding details here
        )
        try:
            # Fetch all the tables from the schema name
            cursor.execute(
                "SELECT table_name from information_schema.tables where table_schema=%s",
                (schema_name,),
            )
            tables_list = cursor.fetchall()

            # Appending table names in the schema info context
            for table in tables_list:
                table_name = table[0]
                schema_info_context = f"{schema_info_context}\nTable: {table_name}\n"

                # Now fetch column names
                # Filter by schema too, otherwise a table with the same name in another schema would mix its columns in here.
                cursor.execute(
                    "SELECT column_name, data_type from information_schema.columns WHERE table_schema = %s AND table_name = %s",
                    (schema_name, table_name),
                )
                columns_list = cursor.fetchall()  # Converts in a list

                # Now add column name and datatype details to schema info context
                for column in columns_list:
                    column_name, data_type = column
                    schema_info_context = (
                        f"{schema_info_context}\nColumn: {column_name}\nData Type: {data_type}"
                    )

                # I do not need all the data, I just need a sample data for my agent/llm to understand how does the data look like.
                # sql.Identifier quotes the schema and table names, so names with mixed case or special characters don't break the query.
                cursor.execute(
                    sql.SQL("SELECT * FROM {}.{} LIMIT 5;").format(
                        sql.Identifier(schema_name), sql.Identifier(table_name)
                    )
                )
                sample_data = cursor.fetchall()  # Converts into the list
                schema_info_context = f"{schema_info_context}\nSample Data:\n"

                # Now, similarly, for the rows.
                for row in sample_data:
                    schema_info_context = f"{schema_info_context} {row}\n"

        except Exception as e:
            print(f"Error handling schema details: {e}")
            schema_info_context = f"Error fetching schema details: {e}"

        finally:
            # Only close the cursor here. Closing the connection would make any later call on this object fail.
            if cursor:
                cursor.close()

        return schema_info_context

    def close(self):
        """
        Close the database connection once you are done with this object.
        """
        if self.connection:
            self.connection.close()

    def execute_sql(self, query):
        connection = self.connection
        cursor = None
        try:
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            connection.commit()
            return str(result)  # Convert the list'd result into string format
        except Exception as e:
            print(f"Error executing query: {e}")
            return f"Error executing query: {e}"
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()


# Lets test the schema
if __name__ == "__main__":
    obj = DatabaseUtils(
        {
            "host": os.environ["host"],
            "port": os.environ["port"],
            "user": os.environ["user"],
            "password": os.environ["password"],
            "dbname": os.environ["database"],
        }
    )

    result = obj.schema_details("public")  # Because public is the schema name.
    obj.close()

    # Now saving it in a text file
    with open("schema_test.txt", "w") as f:
        f.write(result)
