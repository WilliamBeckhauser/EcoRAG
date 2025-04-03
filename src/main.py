import logging
import argparse
from .rag.experiment import Experiment


def setup_logging():
    """
    Sets up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Run RAG experiments with different tasks.'
    )
    parser.add_argument(
        '--schema',
        type=str,
        required=True,
        help='Schema name containing the dataset'
    )
    parser.add_argument(
        '--table',
        type=str,
        required=True,
        help='Table name containing the dataset'
    )
    parser.add_argument(
        '--review-column',
        type=str,
        required=True,
        help='Column name containing the review text'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        help='Column name containing the label (optional)'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='sentiment_analysis',
        choices=[
            'sentiment_analysis',
            'medical_info',
            'review_classification',
            'industrial_machinery',
            'fake_news'
        ],
        help='Task to perform'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='ollama',
        choices=['ollama', 'openai', 'openrouter', 'gemini'],
        help='Provider to use (ollama, openai, openrouter, or gemini)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llama2',
        help='Model name to use (e.g., llama2, gpt-4, etc.)'
    )
    return parser.parse_args()


def main():
    """
    Main entry point for the application.
    """
    # Setup logging
    setup_logging()

    # Parse command line arguments
    args = parse_args()

    try:
        # Initialize and run experiment
        experiment = Experiment(
            schema_name=args.schema,
            table_name=args.table,
            review_column=args.review_column,
            label_column=args.label_column,
            task_name=args.task,
            provider=args.provider,
            model_name=args.model
        )
        experiment.run()

    except Exception as e:
        logging.error(f"Error running experiment: {e}")
        raise


if __name__ == "__main__":
    main()
