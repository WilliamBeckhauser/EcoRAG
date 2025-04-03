import psycopg2
import logging
from typing import List, Dict
from ..config.settings import (
    POSTGRES_DBNAME, POSTGRES_USER, POSTGRES_PASSWORD,
    POSTGRES_HOST, POSTGRES_PORT
)


def get_connection_string() -> str:
    """
    Constructs the PostgreSQL connection string from environment variables.
    """
    return f"dbname='{POSTGRES_DBNAME}' user='{POSTGRES_USER}' password='{POSTGRES_PASSWORD}' host='{POSTGRES_HOST}' port='{POSTGRES_PORT}'"


def load_reviews_dataset_from_postgres(
    conn_str: str,
    schema_name: str,
    table_name: str,
    review_column: str,
    label_column: Optional[str] = None
) -> List[Dict]:
    """
    Loads data from the specified schema and table.
    Expects columns: id, review_column (ex.: title) and optionally
    label_column (classification). Returns a list of dictionaries.
    """
    reviews = []
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()
        if label_column:
            query = f"SELECT id, {review_column}, {label_column} FROM {schema_name}.{table_name};"
        else:
            query = f"SELECT id, {review_column} FROM {schema_name}.{table_name};"
        cursor.execute(query)
        rows = cursor.fetchall()
        for i, row in enumerate(rows):
            dataset_id = row[0] if row[0] is not None else i + 1
            if label_column:
                reviews.append({
                    "dataset_id": dataset_id,
                    "review": row[1],
                    "label": row[2],
                    "source_schema": schema_name
                })
            else:
                reviews.append({
                    "dataset_id": dataset_id,
                    "review": row[1],
                    "label": None,
                    "source_schema": schema_name
                })
        cursor.close()
        conn.close()
        logging.info(
            f"{len(reviews)} records loaded from {schema_name}.{table_name}."
        )
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
    return reviews


def create_or_update_experiment_table(
    conn_str: str,
    experiment_schema: str,
    table_name: str
) -> None:
    """
    Creates or updates the experiment table in PostgreSQL.
    """
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()

        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {experiment_schema}.{table_name} (
            experiment_id TEXT,
            dataset_id INTEGER NOT NULL,
            review TEXT,
            validation TEXT,
            llm_output TEXT,
            chain_of_thought TEXT,
            final_answer TEXT,
            is_correct BOOLEAN,
            latency FLOAT,
            cpu_time FLOAT,
            mem_usage FLOAT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            throughput FLOAT,
            energy_consumption FLOAT,
            flops_per_token FLOAT,
            classification_timestamp TIMESTAMP,
            experiment_date DATE,
            model_name TEXT,
            code_name TEXT,
            prompt TEXT,
            PRIMARY KEY (dataset_id, experiment_id)
        );
        """
        cursor.execute(create_table_query)
        conn.commit()

        alter_queries = [
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS chain_of_thought TEXT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS final_answer TEXT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS throughput FLOAT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS energy_consumption FLOAT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS flops_per_token FLOAT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS experiment_date DATE;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS model_name TEXT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS code_name TEXT;",
            f"ALTER TABLE {experiment_schema}.{table_name} ADD COLUMN IF NOT EXISTS prompt TEXT;"
        ]

        for query in alter_queries:
            cursor.execute(query)

        conn.commit()
        cursor.close()
        conn.close()
        logging.info(
            f"Experiment table {experiment_schema}.{table_name} is ready."
        )

    except Exception as e:
        logging.error(f"Error creating/updating experiment table: {e}")


def insert_experiment_records(
    conn_str: str,
    experiment_schema: str,
    table_name: str,
    records: List[Dict]
) -> None:
    """
    Inserts or updates a batch of records into the experiment table.
    """
    try:
        conn = psycopg2.connect(conn_str)
        cursor = conn.cursor()

        insert_query = f"""
        INSERT INTO {experiment_schema}.{table_name} 
        (experiment_id, dataset_id, review, validation, llm_output, chain_of_thought,
         final_answer, is_correct, latency, cpu_time, mem_usage, 
         prompt_tokens, completion_tokens, total_tokens, throughput, 
         energy_consumption, flops_per_token, classification_timestamp, 
         experiment_date, model_name, code_name, prompt)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (dataset_id, experiment_id) DO UPDATE SET
            review = EXCLUDED.review,
            validation = EXCLUDED.validation,
            llm_output = EXCLUDED.llm_output,
            chain_of_thought = EXCLUDED.chain_of_thought,
            final_answer = EXCLUDED.final_answer,
            is_correct = EXCLUDED.is_correct,
            latency = EXCLUDED.latency,
            cpu_time = EXCLUDED.cpu_time,
            mem_usage = EXCLUDED.mem_usage,
            prompt_tokens = EXCLUDED.prompt_tokens,
            completion_tokens = EXCLUDED.completion_tokens,
            total_tokens = EXCLUDED.total_tokens,
            throughput = EXCLUDED.throughput,
            energy_consumption = EXCLUDED.energy_consumption,
            flops_per_token = EXCLUDED.flops_per_token,
            classification_timestamp = EXCLUDED.classification_timestamp,
            experiment_date = EXCLUDED.experiment_date,
            model_name = EXCLUDED.model_name,
            code_name = EXCLUDED.code_name,
            prompt = EXCLUDED.prompt;
        """

        values = []
        for record in records:
            values.append((
                record["experiment_id"],
                record["dataset_id"],
                record["review"],
                record["validation"],
                record["llm_output"],
                record["chain_of_thought"],
                record["final_answer"],
                record["is_correct"],
                record["latency"],
                record["cpu_time"],
                record["mem_usage"],
                record["prompt_tokens"],
                record["completion_tokens"],
                record["total_tokens"],
                record["throughput"],
                record["energy_consumption"],
                record["flops_per_token"],
                record["classification_timestamp"],
                record["experiment_date"],
                record["model_name"],
                record["code_name"],
                record["prompt"]
            ))

        cursor.executemany(insert_query, values)
        conn.commit()
        cursor.close()
        conn.close()
        logging.info(
            f"{len(records)} records inserted/updated in the experiment table."
        )

    except Exception as e:
        logging.error(f"Error inserting experiment records: {e}")
