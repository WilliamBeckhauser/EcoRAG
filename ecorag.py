from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import json
import random
import csv
import re
import sys
import resource
import logging
import subprocess
import psycopg2
import psutil
import time
import openai
import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Task definitions
TASKS = {
    "sentiment": {
        "name": "Sentiment Analysis",
        "prompt_template": (
            "You are an expert in sentiment analysis. "
            "Each review should be classified into one of five categories based on the sentiment it expresses: "
            "very negative, negative, neutral, positive, or very positive. "
            "Carefully read the review, analyze the tone and language used, and respond with only one of the following labels: "
            "very negative, negative, neutral, positive, or very positive, without any additional words.\n"
            f"Review to be classified: {{text}}."
        ),
        "output_format": "text"
    },
    "medical_extraction": {
        "name": "Medical Information Extraction",
        "prompt_template": (
            "You are an expert in medical information extraction. "
            "Given a medication description, extract the following fields: 'drug' and 'brand'. "
            "Use exactly the same text as it appears in the description, without translating, modifying, or correcting. "
            "If any of the fields are missing or not mentioned, return 'unknown' for that field. "
            "Return the result strictly in JSON format only, without any explanations or extra text.\n\n"
            "Example:\n"
            "Medication description: AC VALPROICO 500MG COMPRIMIDOS BIOLAB\n"
            "Expected output:\n"
            "{\"drug\": \"AC VALPROICO\", \"brand\": \"BIOLAB\"}\n\n"
            f"Medication description: {{text}}\n"
        ),
        "output_format": "json"
    },
    "review_classification": {
        "name": "Review Classification",
        "prompt_template": (
            "Você é um especialista em classificar avaliações. "
            "Essas avaliações podem ser enquadradas em quatro categorias: "
            "Delivery: problemas com entrega ou entregadores. "
            "Quality: questões sobre sabor, ingredientes, forma de preparo etc. "
            "Quantity: quantidade de alimentos ou itens faltantes. "
            "Praise: elogios ou sugestões positivas. "
            "Analise cuidadosamente o conteúdo e responda apenas com uma das categorias: Delivery, Quality, Quantity ou Praise, "
            "sem nenhuma palavra adicional.\n"
            f"Avaliação a ser classificada: {{text}}."
        ),
        "output_format": "text"
    },
    "machine_extraction": {
        "name": "Industrial Machinery Information Extraction",
        "prompt_template": (
            "You are an expert in industrial machinery information extraction. "
            "Given a machine description, extract the following fields: 'brand' and 'model'. "
            "Use exactly the same text as it appears in the description, without translating, modifying, or correcting. "
            "If any of the fields are missing or not mentioned, return 'unknown' for that field. "
            "Return the result strictly in JSON format only, without any explanations or extra text.\n\n"
            "Example:\n"
            "Machine description: John Deere 5075E Utility Tractor\n"
            "Expected output:\n"
            "{\"brand\": \"John Deere\", \"model\": \"5075E\"}\n\n"
            f"Machine description: {{text}}\n"
        ),
        "output_format": "json"
    },
    "fake_news": {
        "name": "Fake News Detection",
        "prompt_template": (
            "Classify the following news as 'real' or 'fake'. "
            "A 'fake' news article may contain false, exaggerated, or misleading information, "
            "often displaying a sensationalist tone, lack of logical support, or internal inconsistencies. "
            "A 'real' news article is coherent, plausible, and free of evident contradictions. "
            "Carefully analyze the content and respond only with 'real' or 'fake', without any additional words. "
            f"News to be classified: {{text}}."
        ),
        "output_format": "text"
    }
}

# Feedback prompts for different tasks
FEEDBACK_PROMPTS = {
    "sentiment": (
        "You are wrong. The correct sentiment is '{correct_answer}'. "
        "Please respond with only the correct sentiment, without any additional words."
    ),
    "medical_extraction": (
        "The correct extraction should be: {correct_answer}. "
        "Please provide the correct JSON output without any additional text."
    ),
    "review_classification": (
        "A classificação correta é '{correct_answer}'. "
        "Por favor, responda apenas com a categoria correta, sem palavras adicionais."
    ),
    "machine_extraction": (
        "The correct extraction should be: {correct_answer}. "
        "Please provide the correct JSON output without any additional text."
    ),
    "fake_news": (
        "The correct classification is '{correct_answer}'. "
        "Please respond with only 'real' or 'fake', without any additional words."
    )
}

# Check if tiktoken is available for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Constant for maximum tokens in memory
MAX_MEMORY_TOKENS = 15000

# Global list to store in-context interactions (questions and answers)
context_memory: List[str] = []

# Initialize Hugging Face embedding model
hf_embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2")

# ------------------------- Helper Functions -------------------------


def approximate_token_count(text: str) -> int:
    """A simple approximation of token count using whitespace splitting."""
    return len(text.split())


def count_tokens(text: str) -> int:
    """
    Counts tokens using tiktoken if available,
    else approximates via whitespace splitting.
    """
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            logging.warning("Error counting tokens with tiktoken: " + str(e))
            return len(text.split())
    else:
        return len(text.split())


def get_context_memory_text(memory: List[str]) -> str:
    """Joins the in-context memory interactions into a single string."""
    return "\n".join(memory)


def update_context_memory(memory: List[str], new_interaction: str) -> List[str]:
    """
    Adiciona uma nova interação à memória e garante que o histórico contenha
    no máximo 10 interações (mantendo as mais recentes).
    """
    memory.append(new_interaction)
    # Limitar a memória para no máximo 10 interações
    while len(memory) > 10:
        memory.pop(0)
    return memory


def separate_thought_and_final(text: str) -> Tuple[str, str]:
    """
    Separates the chain of thought from the final answer if the text
    contains the tags <think> and </think>. Otherwise, returns an empty
    chain and the entire text.
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        chain_of_thought = match.group(1).strip()
        final_answer = re.sub(
            pattern, "", text, flags=re.DOTALL | re.IGNORECASE
        ).strip()
    else:
        chain_of_thought = ""
        final_answer = text.strip()
    return chain_of_thought, final_answer


def initialize_client(api_key: str) -> openai.OpenAI:
    """
    Initializes the OpenAI client with the provided API key.
    """
    openai.api_key = api_key
    return openai


def get_connection_string() -> str:
    """
    Constructs the PostgreSQL connection string from environment variables.
    """
    dbname = os.getenv("POSTGRES_DBNAME")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    return f"dbname='{dbname}' user='{user}' password='{password}' host='{host}' port='{port}'"


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

# ------------------------- Classification Functions -------------------------


def classify_review(
    client: Optional[openai.OpenAI],
    review: str,
    provider: str,
    model_name: str,
    additional_context: str = "",
    task: str = "review_classification"
) -> Tuple[str, float, Optional[float], Optional[float],
           Optional[int], Optional[int], Optional[int]]:
    """
    Routes the classification request to the appropriate provider.
    """
    if provider.lower() == "ollama":
        return ollama_classify(model_name, review, additional_context, task)
    elif provider.lower() == "gemini":
        return gemini_classify(model_name, review, additional_context, task)
    elif provider.lower() == "openrouter":
        return openrouter_classify(client, review, model_name, additional_context, task)
    else:
        return openai_classify(client, review, model_name, additional_context, task)


def openai_classify(
    client: openai.OpenAI,
    review: str,
    model_name: str = "gpt-4o-mini",
    additional_context: str = "",
    task: str = "review_classification"
) -> Tuple[str, float, float, float, Optional[int], Optional[int], Optional[int]]:
    """
    Uses the OpenAI API to classify the text according to the specified task.
    """
    prompt = (additional_context + "\n") if additional_context else ""
    task_template = TASKS[task]["prompt_template"]
    prompt += task_template.format(text=review)

    start_time = time.perf_counter()
    cpu_before = time.process_time()
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024**2)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    end_time = time.perf_counter()
    cpu_after = time.process_time()
    mem_after = proc.memory_info().rss / (1024**2)
    latency = end_time - start_time
    cpu_time = cpu_after - cpu_before
    mem_usage = mem_after - mem_before
    llm_output = completion.choices[0].message.content.strip()

    # For JSON output format, try to parse and validate the JSON
    if TASKS[task]["output_format"] == "json":
        try:
            json.loads(llm_output)
        except json.JSONDecodeError:
            logging.warning(
                f"Invalid JSON output for task {task}: {llm_output}")
            llm_output = "{}"

    try:
        usage = completion["usage"]
    except (TypeError, KeyError):
        usage = {}

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    # Caso não tenha usage do OpenAI, tenta fallback com tiktoken
    if prompt_tokens is None or completion_tokens is None or total_tokens is None:
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                prompt_tokens = len(encoding.encode(prompt))
                completion_tokens = len(encoding.encode(llm_output))
                total_tokens = prompt_tokens + completion_tokens
                logging.info(
                    "Tokens calculated via tiktoken as fallback for OpenAI."
                )
            except Exception as e:
                logging.warning(
                    "Error calculating tokens via tiktoken: " + str(e)
                )
                prompt_tokens, completion_tokens, total_tokens = None, None, None

    return (
        llm_output,
        latency,
        cpu_time,
        mem_usage,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )


def ollama_classify(
    model_name: str,
    review: str,
    additional_context: str = "",
    task: str = "review_classification"
) -> Tuple[str, float, Optional[float], Optional[float],
           Optional[int], Optional[int], Optional[int]]:
    """
    Uses a subprocess call to the Ollama model for classification.
    """
    prompt = (additional_context + "\n") if additional_context else ""
    task_template = TASKS[task]["prompt_template"]
    prompt += task_template.format(text=review)

    start_time = time.perf_counter()
    try:
        process = subprocess.Popen(
            ["ollama", "run", model_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, error = process.communicate(input=prompt, timeout=60)
        process.wait(timeout=10)
    except Exception as e:
        logging.error(f"Error calling Ollama model: {e}")
        return "", 0.0, None, None, None, None, None

    end_time = time.perf_counter()
    latency = end_time - start_time

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_time = usage.ru_utime + usage.ru_stime
    mem_usage = (
        (usage.ru_maxrss / (1024 * 1024))
        if sys.platform == "darwin"
        else usage.ru_maxrss / 1024.0
    )

    llm_output = output.strip() if output else ""

    # For JSON output format, try to parse and validate the JSON
    if TASKS[task]["output_format"] == "json":
        try:
            json.loads(llm_output)
        except json.JSONDecodeError:
            logging.warning(
                f"Invalid JSON output for task {task}: {llm_output}")
            llm_output = "{}"

    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            prompt_tokens = len(encoding.encode(prompt))
            completion_tokens = len(encoding.encode(llm_output))
            total_tokens = prompt_tokens + completion_tokens
            logging.info("Tokens calculated via tiktoken for Ollama.")
        except Exception as e:
            logging.warning(
                "Error calculating tokens via tiktoken for Ollama: " + str(e)
            )
            prompt_tokens, completion_tokens, total_tokens = None, None, None
    else:
        prompt_tokens, completion_tokens, total_tokens = None, None, None

    return (
        llm_output,
        latency,
        cpu_time,
        mem_usage,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )


def gemini_classify(
    model_name: str,
    review: str,
    additional_context: str = "",
    task: str = "review_classification"
) -> Tuple[str, float, Optional[float], Optional[float],
           Optional[int], Optional[int], Optional[int]]:
    """
    Uses the Gemini API to classify the text according to the specified task.
    """
    prompt = (additional_context + "\n") if additional_context else ""
    task_template = TASKS[task]["prompt_template"]
    prompt += task_template.format(text=review)

    start_time = time.perf_counter()
    cpu_before = time.process_time()
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024**2)

    from google import genai
    gemini_client = genai.Client(api_key=os.getenv("API_KEY_GOOGLE"))
    response = gemini_client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    end_time = time.perf_counter()
    cpu_after = time.process_time()
    mem_after = proc.memory_info().rss / (1024**2)

    latency = end_time - start_time
    cpu_time = cpu_after - cpu_before
    mem_usage = mem_after - mem_before

    llm_output = response.text.strip() if response.text else ""

    # For JSON output format, try to parse and validate the JSON
    if TASKS[task]["output_format"] == "json":
        try:
            json.loads(llm_output)
        except json.JSONDecodeError:
            logging.warning(
                f"Invalid JSON output for task {task}: {llm_output}")
            llm_output = "{}"

    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            prompt_tokens = len(encoding.encode(prompt))
            completion_tokens = len(encoding.encode(llm_output))
            total_tokens = prompt_tokens + completion_tokens
            logging.info("Tokens calculated via tiktoken for Gemini.")
        except Exception as e:
            logging.warning(
                "Error calculating tokens via tiktoken for Gemini: " + str(e)
            )
            prompt_tokens, completion_tokens, total_tokens = None, None, None
    else:
        prompt_tokens, completion_tokens, total_tokens = None, None, None

    return (
        llm_output,
        latency,
        cpu_time,
        mem_usage,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )


def openrouter_classify(
    client: Any,
    review: str,
    model_name: str,
    additional_context: Optional[str] = None,
    task: Optional[str] = None
) -> Tuple[str, float, float, float, int, int, int]:
    """
    Uses the OpenRouter API (via the OpenAI-compatible interface)
    to classify a review based on the specified task.
    """
    start_time = time.perf_counter()
    cpu_before = time.process_time()
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / (1024**2)

    try:
        # Construct the prompt based on the task
        task_template = TASKS[task]["prompt_template"]
        prompt = task_template.format(text=review)
        if additional_context:
            prompt = f"{additional_context}\n{prompt}"

        # Make the API call
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )

        end_time = time.perf_counter()
        cpu_after = time.process_time()
        mem_after = proc.memory_info().rss / (1024**2)

        latency = end_time - start_time
        cpu_time = cpu_after - cpu_before
        mem_usage = mem_after - mem_before

        llm_output = response.choices[0].message.content.strip()

        # For JSON output format, try to parse and validate the JSON
        if TASKS[task]["output_format"] == "json":
            try:
                json.loads(llm_output)
            except json.JSONDecodeError:
                logging.warning(
                    f"Invalid JSON output for task {task}: {llm_output}")
                llm_output = "{}"

        return (
            llm_output,
            latency,
            cpu_time,
            mem_usage,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens
        )

    except Exception as e:
        logging.error(f"Error calling OpenRouter API: {e}")
        return ("error", 0.0, 0.0, 0.0, 0, 0, 0)

# ------------------------- Graph-RAG Functions -------------------------


def get_embedding(text: str, dim: int = 768) -> List[float]:
    """
    Generates an embedding for the provided text using the
    Hugging Face model (SentenceTransformer).
    """
    try:
        embedding = hf_embedding_model.encode(text).tolist()
        if not embedding:
            logging.error("Empty embedding generated by HF model.")
            return [0.0] * dim
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding with HF model: {e}")
        return [0.0] * dim


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Computes the cosine similarity between two vectors.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_context(
    review_text: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    top_n: int = 3
) -> str:
    """
    Graph retrieval approach:
    1. Converte o texto completo da notícia em um embedding.
    2. Consulta o banco Neo4j para nós :News (title, classification, embedding).
    3. Calcula similaridade (cosine) entre o embedding do texto e cada embedding no grafo.
    4. Seleciona os top_n nós mais similares.
    5. Retorna um contexto que lista exemplos correspondentes.
    """
    # 1. Converte o texto inteiro em embedding
    text_embedding = get_embedding(review_text)

    # 2. Consulta apenas nós :News no Neo4j
    context_items = []
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    with driver.session() as session:
        result_news = session.run(
            "MATCH (n:News) RETURN n.title AS title, "
            "n.classification AS classification, n.embedding AS embedding"
        )
        for record in result_news:
            title = record["title"]
            classification = record["classification"]
            embedding = record["embedding"]
            if not embedding:
                continue
            # 3. Similaridade entre o texto e cada embedding
            sim = cosine_similarity(text_embedding, embedding)
            context_items.append((sim, title, classification))

    driver.close()

    # 4. Seleciona top_n nós mais similares
    context_items.sort(key=lambda x: x[0], reverse=True)
    top_items = context_items[:top_n]

    # 5. Monta a string de contexto
    context_str = ""
    for _, title, classification in top_items:
        context_str += f"Example of {classification} review: {title}\n"

    return context_str

# ------------------------- Experiment and CSV Functions -------------------------


def export_records_to_csv(records: List[Dict], filename: str) -> None:
    """
    Exports records to a CSV file.
    """
    if not records:
        logging.warning("No records to export.")
        return
    headers = records[0].keys()
    try:
        with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for record in records:
                writer.writerow(record)
        logging.info(f"Records exported to CSV: {filename}")
    except Exception as e:
        logging.error(f"Error exporting to CSV: {e}")


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


# ------------------------- Main Function -------------------------


def main() -> None:
    global context_memory  # Declarar como global para atualizar a memória de contexto

    parser = argparse.ArgumentParser(
        description=(
            "Processa experimentos de classificação via LLM com In-context learning e Graph-RAG. "
            "Para cada notícia, recupera contexto do grafo (via similaridade), "
            "constrói o prompt e classifica a notícia como 'real' ou 'fake'."
        )
    )
    parser.add_argument("--source_schema", type=str,
                        required=True, help="Schema da fonte (ex.: test)")
    parser.add_argument("--table_name", type=str, required=True,
                        help="Nome da tabela fonte (ex.: fake_news_gossipcop)")
    parser.add_argument("--review_column", type=str, required=True,
                        help="Nome da coluna de review (ex.: title)")
    parser.add_argument("--label_column", type=str, required=False, default=None,
                        help="Nome da coluna de validação (opcional, ex.: classification)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Nome do modelo LLM (ex.: gpt-4o-mini, ollama, openrouter ou gemini)")
    parser.add_argument("--code_name", type=str, required=False,
                        default="rag", help="Nome do código (ex.: rag)")
    parser.add_argument("--provider", type=str, required=False, default="openai",
                        choices=["openai", "ollama", "openrouter", "gemini"],
                        help="Provider a ser usado (openai, ollama, openrouter ou gemini)")
    parser.add_argument("--api_key", type=str, required=False,
                        help="API key para o provider (se não fornecido, será lido do .env)")
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASKS.keys()),
                        help="Task to be performed (sentiment, medical_extraction, review_classification, machine_extraction, fake_news)")
    # Credenciais Neo4j
    parser.add_argument("--neo4j_uri", type=str,
                        default=os.getenv("NEO4J_URI"),
                        help="URI do Neo4j (se não fornecido, será lido do .env)")
    parser.add_argument("--neo4j_user", type=str,
                        default=os.getenv("NEO4J_USER"),
                        help="Usuário do Neo4j (se não fornecido, será lido do .env)")
    parser.add_argument("--neo4j_password", type=str,
                        default=os.getenv("NEO4J_PASSWORD"),
                        help="Senha do Neo4j (se não fornecido, será lido do .env)")
    # Novo argumento para modo de execução
    parser.add_argument("--execution_mode", type=str, default="ecorag",
                        choices=["base", "rag", "icl", "ecorag"],
                        help=(
                            "Modo de execução: 'base' (sem RAG ou ICL), "
                            "'rag' (apenas RAG), 'icl' (apenas memória) ou 'ecorag' (RAG e ICL)"
                        ))
    args = parser.parse_args()

    # Escolha do provider
    if args.provider.lower() == "openai":
        api_key = args.api_key if args.api_key else os.getenv("API_KEY_OPENAI")
        if not api_key:
            logging.error(
                "API_KEY_OPENAI não fornecido ou não encontrado no .env para OpenAI")
            return
        openai.api_key = api_key
        client = openai  # agora client é o próprio módulo openai
    elif args.provider.lower() == "openrouter":
        if not os.getenv("OPENROUTER_TOKEN"):
            raise ValueError(
                "OPENROUTER_TOKEN não fornecido ou não encontrado no .env para OpenRouter")
        client = openai.OpenAI(
            api_key=os.getenv("OPENROUTER_TOKEN"), base_url="https://openrouter.ai/api/v1"
        )
    elif args.provider.lower() == "gemini":
        if not os.getenv("API_KEY_GOOGLE"):
            logging.error("API_KEY_GOOGLE não encontrado no .env para Gemini")
            return
        client = None
    elif args.provider.lower() == "ollama":
        client = None
    else:
        client = None

    # Verifica se as credenciais do Neo4j estão disponíveis
    if not args.neo4j_uri or not args.neo4j_user or not args.neo4j_password:
        logging.error("Credenciais do Neo4j não encontradas no .env")
        return

    conn_str = get_connection_string()

    # Carrega os datasets de teste e validação conforme o modo de execução
    test_dataset = load_reviews_dataset_from_postgres(
        conn_str, args.source_schema, args.table_name,
        args.review_column, args.label_column
    )
    validator_dataset = load_reviews_dataset_from_postgres(
        conn_str, "validator", args.table_name,
        args.review_column, args.label_column
    )

    if args.execution_mode in ["icl", "ecorag"]:
        if len(validator_dataset) >= 50:
            first_ten_validator = validator_dataset[:50]
            remaining_validator = validator_dataset[50:]
        else:
            first_ten_validator = validator_dataset
            remaining_validator = []
        combined_remaining = test_dataset + remaining_validator
        random.shuffle(combined_remaining)
        reviews_dataset = first_ten_validator + combined_remaining
        logging.info(
            f"Total de registros após mesclagem (ICL/EcoRAG): {len(reviews_dataset)}"
        )
    else:
        reviews_dataset = test_dataset
        logging.info(f"Total de registros (Base/RAG): {len(reviews_dataset)}")

    # Cria ou atualiza a tabela de experimentos
    experiment_schema = "experiments"
    create_or_update_experiment_table(
        conn_str, experiment_schema, args.table_name)

    batch_size = 2
    batch_records = []
    all_records = []

    # Estimativa de FLOPs do modelo (valor hipotético)
    model_flops_estimate = 1e11

    for i, review_data in enumerate(reviews_dataset, start=1):
        dataset_id = review_data["dataset_id"]
        review_text = review_data["review"]
        validation = review_data["label"]

        # Construir o contexto de acordo com o modo de execução
        if args.execution_mode == "base":
            combined_context = ""
        elif args.execution_mode == "rag":
            combined_context = retrieve_context(
                review_text, args.neo4j_uri, args.neo4j_user,
                args.neo4j_password, top_n=5
            )
        elif args.execution_mode == "icl":
            combined_context = get_context_memory_text(context_memory)
        elif args.execution_mode == "ecorag":
            graph_context = retrieve_context(
                review_text, args.neo4j_uri, args.neo4j_user,
                args.neo4j_password, top_n=5
            )
            mem_context = get_context_memory_text(context_memory)
            combined_context = (mem_context + "\n" +
                                graph_context) if mem_context else graph_context
        else:
            combined_context = ""

        # Construir o prompt inicial (sem tentativas de correção)
        initial_prompt = (combined_context + "\n") if combined_context else ""
        task_template = TASKS[args.task]["prompt_template"]
        initial_prompt += task_template.format(text=review_text)

        # Primeira chamada à classificação
        (
            llm_output,
            latency,
            cpu_time,
            mem_usage,
            prompt_tokens,
            completion_tokens,
            total_tokens
        ) = classify_review(
            client,
            review_text,
            args.provider,
            args.model_name,
            combined_context,
            args.task
        )

        # Separa cadeia de pensamento (<think> ... </think>) do resultado final
        chain_of_thought, final_answer = separate_thought_and_final(llm_output)

        # Avalia se está correto (caso haja rótulo)
        if TASKS[args.task]["output_format"] == "json":
            try:
                # For JSON output, compare the parsed JSON with the validation
                if validation is not None:
                    try:
                        validation_json = json.loads(validation)
                        output_json = json.loads(final_answer)
                        is_correct = validation_json == output_json
                    except json.JSONDecodeError:
                        is_correct = False
                else:
                    is_correct = None
            except json.JSONDecodeError:
                is_correct = False
        else:
            # For text output, compare strings directly
            is_correct = (
                final_answer.lower() == validation.lower()
                if validation is not None
                else None
            )

        attempts = 1  # Contador de tentativas

        # Se for um registro do schema "validator" e houver rótulo,
        # podemos tentar corrigir a resposta com prompt adicional
        if review_data.get("source_schema", "test").lower() == "validator" and validation is not None:
            while not is_correct and attempts <= 5:
                reattempt_message = FEEDBACK_PROMPTS[args.task].format(
                    correct_answer=validation
                )

                combined_context_retry = combined_context + "\n" + reattempt_message
                logging.info(
                    f"Attempt {attempts} para dataset_id {dataset_id}")

                # Nova tentativa
                (
                    retry_output,
                    retry_latency,
                    retry_cpu_time,
                    retry_mem_usage,
                    retry_prompt_tokens,
                    retry_completion_tokens,
                    retry_total_tokens
                ) = classify_review(
                    client,
                    review_text,
                    args.provider,
                    args.model_name,
                    combined_context_retry,
                    args.task
                )

                chain_of_thought, retry_final_answer = separate_thought_and_final(
                    retry_output)

                # Avalia se está correto (caso haja rótulo)
                if TASKS[args.task]["output_format"] == "json":
                    try:
                        # For JSON output, compare the parsed JSON with the validation
                        validation_json = json.loads(validation)
                        output_json = json.loads(retry_final_answer)
                        is_correct = validation_json == output_json
                    except json.JSONDecodeError:
                        is_correct = False
                else:
                    # For text output, compare strings directly
                    is_correct = (retry_final_answer.lower()
                                  == validation.lower())

                final_answer = retry_final_answer
                llm_output = retry_output
                attempts += 1

                if is_correct:
                    logging.info(
                        f"Resposta correta obtida na tentativa {attempts-1} "
                        f"para dataset_id {dataset_id}"
                    )
                    break

                if attempts > 5:
                    logging.info(
                        f"Após 5 tentativas, resposta final (incorreta) "
                        f"para dataset_id {dataset_id} foi '{final_answer}'"
                    )
                    break

        classification_timestamp = datetime.now()
        experiment_date = classification_timestamp.date()
        experiment_id = f"{args.table_name}_{args.code_name}_{args.model_name}_{args.code_name}"

        # Throughput e métricas adicionais
        throughput = (
            (total_tokens / latency)
            if (latency > 0 and total_tokens is not None)
            else None
        )
        energy_consumption = (
            (cpu_time * 50) if cpu_time is not None else None
        )
        flops_per_token = (
            (model_flops_estimate / total_tokens)
            if (total_tokens and total_tokens > 0)
            else None
        )

        record = {
            "experiment_id": experiment_id,
            "dataset_id": dataset_id,
            "review": review_text,
            "validation": validation,
            "llm_output": llm_output,
            "chain_of_thought": chain_of_thought,
            "final_answer": final_answer,
            "is_correct": is_correct,
            "latency": latency,
            "cpu_time": cpu_time,
            "mem_usage": mem_usage,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "throughput": throughput,
            "energy_consumption": energy_consumption,
            "flops_per_token": flops_per_token,
            "classification_timestamp": classification_timestamp,
            "experiment_date": experiment_date,
            "model_name": args.model_name,
            "code_name": args.code_name,
            "prompt": initial_prompt
        }

        batch_records.append(record)
        all_records.append(record)

        # Atualiza a memória de contexto (ICL/EcoRAG)
        if args.execution_mode in ["icl", "ecorag"]:
            if (review_data.get("source_schema", "test").lower() == "validator"
                    and validation is not None
                    and attempts > 1):
                interaction = f'Exemplo of {validation}: "{review_text}; "'
                context_memory = update_context_memory(
                    context_memory, interaction)

        # Insere em lote
        if i % batch_size == 0:
            insert_experiment_records(
                conn_str, experiment_schema, args.table_name, batch_records
            )
            logging.info(
                f"Lote de {batch_size} registros inserido no banco de dados."
            )
            batch_records.clear()
            logging.info("Aguardando 15 segundos antes do próximo lote...")
            time.sleep(5)

    # Insere resto do lote
    if batch_records:
        insert_experiment_records(
            conn_str, experiment_schema, args.table_name, batch_records
        )
        logging.info(
            f"Lote final de {len(batch_records)} registros inserido no banco de dados."
        )


if __name__ == "__main__":
    main()

# python ecorag.py --source_schema test --table_name politifact --review_column title --label_column label --model_name llama3.2:1b --provider ollama --code_name ecorag --execution_mode ecorag --task sentiment
