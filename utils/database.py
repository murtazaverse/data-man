"""
Here we will create a class with which we can tell
how to connect with the server and how to fetch the data.
"""

import psycopg2

class DatabaseUtils:
    """
    Connection to the database.
    """

    def __init__(self, db_config):
        self.db_config=db_config

        try:
            self.connection=psycopg2.connect(**db_config)
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            self.connection=None

    def schema_details(self, schema_name):
        """
        Query to find the context from my database.
        I am preparing the context here for the LLM.
        (Use RAG here later)
        """

        schema_info_context = ""

        conn = self.connection
        cursor = conn.cursor() # The object that we get from the postgre connection, with the  help of it we can run the queries.

        schema_info_context = f"Database Schema: {schema_name}\n" # We will keep on adding details here

        # Fetch all the tables from the schema name
        cursor.execute("SELECT table_name from information_schema.tables where tabel_schema=%s",(schema_name,))
        tables_list=cursor.fetchall()

        # Appending table names in the schema info context
        for table in tables_list:
            table_name = table[0]
            schema_info_context = f"{schema_info_context}\nTable: {table_name}\n"

            # Now fetch column names
            cursor.execute("SELECT column_name, data_type from information_schema.columns WHERE table_name = %s", (table_name,))
            columns_list = cursor.fetchall() # Converts in a list

            # Now add column name and datatype details to schema info context
            for column in column_list:
                column_name, data_type = column
                schema_info_context = f"{schema_info_context}\nColumn: {column_name}\nData Type: {data_type}"

                # Continue from 1:47 