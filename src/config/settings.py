import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
API_KEY_OPENAI = os.getenv("API_KEY_OPENAI")
API_KEY_GOOGLE = os.getenv("API_KEY_GOOGLE")
OPEN_ROOT = os.getenv("OPEN_ROOT")
OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN")

# PostgreSQL settings
POSTGRES_DBNAME = os.getenv("POSTGRES_DBNAME")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

# Neo4j settings
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Task definitions
TASKS = {
    "sentiment_analysis": {
        "name": "sentiment_analysis",
        "prompt_template": (
            "You are an expert in sentiment analysis. "
            "Each review should be classified into one of five categories based on the sentiment it expresses: "
            "very negative, negative, neutral, positive, or very positive. "
            "Carefully read the review, analyze the tone and language used, and respond with only one of the following labels: "
            "very negative, negative, neutral, positive, or very positive, without any additional words.\n"
            f"Review to be classified: {{input}}."
        ),
        "feedback_prompt": (
            "You are wrong. The correct sentiment is '{correct_answer}'. "
            "Please respond with only the correct sentiment, without any additional words."
        )
    },
    "medical_info": {
        "name": "medical_info",
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
            f"Medication description: {{input}}\n"
        ),
        "feedback_prompt": (
            "The correct extraction should be: {correct_answer}. "
            "Please provide the correct JSON output without any additional text."
        )
    },
    "review_classification": {
        "name": "review_classification",
        "prompt_template": (
            "Você é um especialista em classificar avaliações. "
            "Essas avaliações podem ser enquadradas em quatro categorias: "
            "Delivery: problemas com entrega ou entregadores. "
            "Quality: questões sobre sabor, ingredientes, forma de preparo etc. "
            "Quantity: quantidade de alimentos ou itens faltantes. "
            "Praise: elogios ou sugestões positivas. "
            "Analise cuidadosamente o conteúdo e responda apenas com uma das categorias: Delivery, Quality, Quantity ou Praise, "
            "sem nenhuma palavra adicional.\n"
            f"Avaliação a ser classificada: {{input}}."
        ),
        "feedback_prompt": (
            "A classificação correta é '{correct_answer}'. "
            "Por favor, responda apenas com a categoria correta, sem palavras adicionais."
        )
    },
    "industrial_machinery": {
        "name": "industrial_machinery",
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
            f"Machine description: {{input}}\n"
        ),
        "feedback_prompt": (
            "The correct extraction should be: {correct_answer}. "
            "Please provide the correct JSON output without any additional text."
        )
    },
    "fake_news": {
        "name": "fake_news",
        "prompt_template": (
            "Classify the following news as 'real' or 'fake'. "
            "A 'fake' news article may contain false, exaggerated, or misleading information, "
            "often displaying a sensationalist tone, lack of logical support, or internal inconsistencies. "
            "A 'real' news article is coherent, plausible, and free of evident contradictions. "
            "Carefully analyze the content and respond only with 'real' or 'fake', without any additional words. "
            f"News to be classified: {{input}}."
        ),
        "feedback_prompt": (
            "The correct classification is '{correct_answer}'. "
            "Please respond with only 'real' or 'fake', without any additional words."
        )
    }
}

# Constants
MAX_MEMORY_TOKENS = 15000
BATCH_SIZE = 2
MODEL_FLOPS_ESTIMATE = 1e11
