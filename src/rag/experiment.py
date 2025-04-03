import logging
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any
from ..config.settings import (
    TASKS, EXPERIMENT_SCHEMA, EXPERIMENT_TABLE,
    BATCH_SIZE
)
from ..utils.database import (
    get_connection_string,
    load_reviews_dataset_from_postgres,
    create_or_update_experiment_table,
    insert_experiment_records
)
from ..utils.performance import (
    get_process_metrics,
    measure_performance,
    log_performance_metrics
)
from ..providers.llm_provider import LLMProvider


class Experiment:
    def __init__(
        self,
        schema_name: str,
        table_name: str,
        review_column: str,
        label_column: Optional[str] = None,
        task_name: str = "sentiment_analysis",
        provider: str = "ollama",
        model_name: str = "llama2"
    ):
        self.schema_name = schema_name
        self.table_name = table_name
        self.review_column = review_column
        self.label_column = label_column
        self.task_name = task_name
        self.task_definition = TASKS[task_name]
        self.llm_provider = LLMProvider(
            provider=provider, model_name=model_name)
        self.conn_str = get_connection_string()

        # Initialize experiment table
        create_or_update_experiment_table(
            self.conn_str,
            EXPERIMENT_SCHEMA,
            EXPERIMENT_TABLE
        )

    def run(self) -> None:
        """
        Runs the experiment for the specified task.
        """
        # Load dataset
        reviews = load_reviews_dataset_from_postgres(
            self.conn_str,
            self.schema_name,
            self.table_name,
            self.review_column,
            self.label_column
        )

        if not reviews:
            logging.error("No reviews loaded from the database.")
            return

        # Process reviews in batches
        for i in range(0, len(reviews), BATCH_SIZE):
            batch = reviews[i:i + BATCH_SIZE]
            self._process_batch(batch)

        logging.info("Experiment completed successfully.")

    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """
        Processes a batch of reviews.
        """
        records = []

        for review in batch:
            start_time = time.time()
            initial_metrics = get_process_metrics()

            # Generate response
            response = self.llm_provider.generate_response(
                review["review"],
                self.task_name,
                self.task_definition
            )

            # Validate response
            is_valid = self.llm_provider.validate_response(
                response,
                self.task_definition
            )

            # Get final metrics
            final_metrics = get_process_metrics()
            performance = measure_performance(
                start_time,
                response.get("total_tokens", 0),
                final_metrics["cpu_time"],
                final_metrics["mem_usage"]
            )

            # Log performance metrics
            log_performance_metrics(performance)

            # Prepare record
            record = {
                "experiment_id": str(uuid.uuid4()),
                "dataset_id": review["dataset_id"],
                "review": review["review"],
                "validation": str(is_valid),
                "llm_output": str(response),
                "chain_of_thought": response.get("chain_of_thought", ""),
                "final_answer": str(response.get("final_answer", "")),
                "is_correct": is_valid,
                "latency": performance["latency"],
                "cpu_time": performance["cpu_time"],
                "mem_usage": performance["mem_usage"],
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
                "total_tokens": response.get("total_tokens", 0),
                "throughput": performance["throughput"],
                "energy_consumption": performance["energy_consumption"],
                "flops_per_token": performance["flops_per_token"],
                "classification_timestamp": datetime.now(),
                "experiment_date": datetime.now().date(),
                "model_name": self.llm_provider.model_name,
                "code_name": "ecorag",
                "prompt": response.get("prompt", "")
            }

            records.append(record)

        # Insert records into database
        insert_experiment_records(
            self.conn_str,
            EXPERIMENT_SCHEMA,
            EXPERIMENT_TABLE,
            records
        )
