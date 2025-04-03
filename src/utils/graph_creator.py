import os
import psycopg2
import spacy
import logging
import argparse
import subprocess
import json
from neo4j import GraphDatabase
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_postgres_connection() -> psycopg2.extensions.connection:
    """
    Establishes and returns a PostgreSQL connection using credentials from the environment.
    """
    dbname = os.getenv("POSTGRES_DBNAME")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    conn_str = f"dbname='{dbname}' user='{user}' password='{password}' host='{host}' port='{port}'"
    return psycopg2.connect(conn_str)


def load_data_from_postgres(
    schema: str,
    table: str,
    id_column: str,
    title_column: str,
    classification_column: str
) -> List[Dict[str, Any]]:
    """
    Loads data from the specified schema and table in PostgreSQL.
    Expects columns defined by the provided column names.

    Args:
        schema (str): The schema name.
        table (str): The table name.
        id_column (str): The column name for the ID.
        title_column (str): The column name for the title.
        classification_column (str): The column name for the classification.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the rows.
    """
    conn = get_postgres_connection()
    cursor = conn.cursor()
    query = f"""
        SELECT {id_column}, {title_column}, {classification_column}
        FROM {schema}.{table}
        LIMIT 3000;
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    data = []
    for row in rows:
        data.append({
            "id": row[0],
            "title": row[1],
            "classification": row[2]
        })
    cursor.close()
    conn.close()
    return data


def get_topic_from_llama(text: str, model_name: str = "llama3.2:1b") -> str:
    """
    Sends the news headline to the Llama3:8b model via Ollama to extract the main topic.

    Args:
        text (str): The text to analyze.
        model_name (str): The model name to use (default "llama3:8b").

    Returns:
        str: The extracted main topic, or "undefined" if extraction fails.
    """
    prompt = f"Extract the main topic of the following news headline: {text}. Respond only with the main topic, without any additional text."
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
        topic = output.strip()
        if not topic:
            logging.error(f"Empty topic for headline: {text}")
            return "undefined"
        return topic
    except Exception as e:
        logging.error(f"Error extracting topic via Llama3:8b: {e}")
        return "undefined"


def get_entities_from_llama(text: str, model_name: str = "llama3:8b") -> List[str]:
    """
    Sends the news headline to the Llama3:8b model via Ollama to extract up to 2 key entities.

    Args:
        text (str): The text to analyze.
        model_name (str): The model name to use (default "llama3:8b").

    Returns:
        List[str]: A list of extracted entities, or an empty list if extraction fails.
    """
    prompt = f"Extract up to 2 key entities from the following news headline: {text}. Respond with a JSON array containing the entities."
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
        output = output.strip()
        if not output:
            logging.error(f"Empty entity extraction for headline: {text}")
            return []
        try:
            entities = json.loads(output)
            if not isinstance(entities, list):
                logging.error(
                    f"Entity extraction output is not a list: {output}")
                return []
            return entities
        except Exception as e:
            entities = [ent.strip()
                        for ent in output.split(',') if ent.strip()]
            return entities
    except Exception as e:
        logging.error(f"Error extracting entities via Llama3:8b: {e}")
        return []


embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_embedding_hf(text: str, dim: int = 768) -> List[float]:
    """
    Generates an embedding for the given text using a Hugging Face model.

    Args:
        text (str): The text to encode.
        dim (int): The expected dimension of the embedding (default 768).

    Returns:
        List[float]: The generated embedding as a list of floats, or a zero vector if an error occurs.
    """
    try:
        embedding = embedding_model.encode(text).tolist()
        if not embedding:
            logging.error("Empty embedding generated by Hugging Face model.")
            return [0.0] * dim
        return embedding
    except Exception as e:
        logging.error(
            f"Error generating embedding with Hugging Face model: {e}")
        return [0.0] * dim


class Neo4jConnector:
    """
    Connector class for interacting with a Neo4j database.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        """
        Initializes the Neo4jConnector with the provided URI, user, and password.

        Args:
            uri (str): The Neo4j connection URI.
            user (str): The Neo4j username.
            password (str): The Neo4j password.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """
        Closes the Neo4j driver connection.
        """
        self.driver.close()

    def create_schema_nodes_and_relations(self, news_data: List[Dict[str, Any]]) -> None:
        """
        Creates nodes and relationships in Neo4j based on the provided news data.

        Nodes:
            - :News with properties {title, classification, embedding}
            - :Topic with properties {description, embedding}
            - :Entity with properties {name, embedding}

        Relationships:
            - (News)-[:HAS_TOPIC]->(Topic)
            - (News)-[:CONTAINS]->(Entity)
            - (Entity)-[:SIMILAR_TO]->(Entity) among entities extracted from the same news.

        Args:
            news_data (List[Dict[str, Any]]): A list of news data dictionaries.
        """
        with self.driver.session() as session:
            for item in news_data:
                title = item["title"]
                classification = item["classification"]

                # Create :News node with title and classification
                session.run("""
                    MERGE (n:News {title: $title})
                    SET n.classification = $classification
                """, title=title, classification=classification)

                # Generate embedding for News using Hugging Face
                news_embedding = get_embedding_hf(title)
                session.run("""
                    MATCH (n:News {title: $title})
                    SET n.embedding = $embedding
                """, title=title, embedding=news_embedding)

                # Extract main topic using Llama3:8b
                topic_principal = get_topic_from_llama(
                    title, model_name="llama3:8b")
                if topic_principal == "":
                    logging.error(f"Empty topic for headline: {title}")
                    topic_principal = "undefined"

                # Generate embedding for Topic using Hugging Face
                topic_embedding = get_embedding_hf(topic_principal)
                session.run("""
                    MERGE (t:Topic {description: $desc})
                    SET t.embedding = $embedding
                """, desc=topic_principal, embedding=topic_embedding)

                # Create relationship: (News)-[:HAS_TOPIC]->(Topic)
                session.run("""
                    MATCH (n:News {title: $title}), (t:Topic {description: $desc})
                    MERGE (n)-[:HAS_TOPIC]->(t)
                """, title=title, desc=topic_principal)

                # Extract entities using Llama3:8b
                entities = get_entities_from_llama(
                    title, model_name="llama3.2:1b")
                for ent in entities:
                    ent_embedding = get_embedding_hf(ent)
                    session.run("""
                        MERGE (e:Entity {name: $ent})
                        SET e.embedding = $embedding
                    """, ent=ent, embedding=ent_embedding)
                    session.run("""
                        MATCH (n:News {title: $title}), (e:Entity {name: $ent})
                        MERGE (n)-[:CONTAINS]->(e)
                    """, title=title, ent=ent)

                # Create SIMILAR_TO relationships among extracted entities
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        ent1 = entities[i]
                        ent2 = entities[j]
                        session.run("""
                            MATCH (e1:Entity {name: $ent1}), (e2:Entity {name: $ent2})
                            MERGE (e1)-[:SIMILAR_TO]->(e2)
                            MERGE (e2)-[:SIMILAR_TO]->(e1)
                        """, ent1=ent1, ent2=ent2)


def main() -> None:
    """
    Main function for graph creation.

    Loads data from PostgreSQL based on the provided schema, table, and column names,
    extracts topics and entities using Llama3:8b via Ollama,
    generates embeddings using a Hugging Face model, and
    creates nodes and relationships in Neo4j.

    Command-line arguments:
        --neo4j_uri: Neo4j connection URI (default from .env or provided).
        --neo4j_user: Neo4j username.
        --neo4j_password: Neo4j password.
        --postgres_schema: PostgreSQL schema name for graph data.
        --postgres_table: PostgreSQL table name for graph data.
        --id_column: Column name for the ID (default "id").
        --title_column: Column name for the title (default "title").
        --classification_column: Column name for the classification (default "classification").
    """
    parser = argparse.ArgumentParser(
        description="Creates a knowledge graph with :News nodes (title, classification, embedding), "
                    ":Topic nodes (description, embedding), and :Entity nodes (name, embedding). "
                    "Relationships: News HAS_TOPIC Topic, News CONTAINS Entity, Entity SIMILAR_TO Entity."
    )
    parser.add_argument("--neo4j_uri", default=os.getenv("NEO4J_URI",
                        "neo4j+s://80380e7f.databases.neo4j.io"))
    parser.add_argument(
        "--neo4j_user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument(
        "--neo4j_password", default=os.getenv("NEO4J_PASSWORD", "your_default_neo4j_password"))
    parser.add_argument("--postgres_schema",
                        default=os.getenv("GRAPH_POSTGRES_SCHEMA", "graph"))
    parser.add_argument(
        "--postgres_table", default=os.getenv("GRAPH_POSTGRES_TABLE", "fake_news_gossipcop"))
    parser.add_argument("--id_column", default="id",
                        help="Column name for the ID (default: id)")
    parser.add_argument("--title_column", default="title",
                        help="Column name for the title (default: title)")
    parser.add_argument("--classification_column", default="classification",
                        help="Column name for the classification (default: classification)")
    args = parser.parse_args()

    logging.info(
        "Loading data from PostgreSQL (only title and classification)...")
    data = load_data_from_postgres(args.postgres_schema, args.postgres_table,
                                   args.id_column, args.title_column, args.classification_column)
    logging.info(f"Total records loaded: {len(data)}")

    logging.info(
        "Loading spaCy model (en_core_web_sm) for NER (not used in this example)...")
    nlp_model = spacy.load("en_core_web_sm")

    logging.info("Connecting to Neo4j and creating the graph...")
    connector = Neo4jConnector(
        args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    connector.create_schema_nodes_and_relations(data)
    connector.close()

    logging.info(
        "Graph creation completed with :News, :Topic, and :Entity nodes and their relationships.")


if __name__ == "__main__":
    main()

# python src/utils/graph_creator.py --neo4j_uri "neo4j+s://b59cc8c2.databases.neo4j.io" --neo4j_user "neo4j" --neo4j_password "jTMc2sLu4xEmv5G46B8uwzqXxO_Fltwld-rEGjk-ewM" --postgres_schema "graph" --postgres_table "politifact" --id_column "id" --title_column "title" --classification_column "label"

# python graph_creator.py --neo4j_uri "your_neo4j_uri" --neo4j_user "your_neo4j_user" --neo4j_password "your_neo4j_password" --postgres_schema "graph" --postgres_table "fake_news_gossipcop"
