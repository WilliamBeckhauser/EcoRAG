# EcoRAG

**EcoRAG**
The EcoRAG framework, combines RAG through KGs (Graph-RAG) with an iterative validation mechanism based on ICL. The method is structured into three main phases: (i) Pre-production: responsible for organizing and structuring domain knowledge; (ii) Prompt Raising: which builds enriched prompts from the KG; and (iii) ICL-Based Validation: where an automated module performs checks and issues corrective textual feedback. This iterative validation is integrated into the model’s contextual memory, allowing for successive improvements in the responses without changing the LLM’s parameters or resorting to fine-tuning techniques.


---

## ⚙️ How It Works

### 1. Task Processing Workflow

1. **Data Loading**: Retrieves data from a PostgreSQL database.
2. **Task Execution**:
   - Applies task-specific prompt templates.
   - Sends requests to the language model.
   - Validates responses based on task criteria.
3. **Performance Monitoring**:
   - Measures latency, throughput, token usage.
   - Monitors CPU, memory, and energy usage.
   - Calculates FLOPS per token.
4. **Result Storage**: Saves all metrics and responses into PostgreSQL for later analysis.

### 2. Supported Tasks

- **Sentiment Analysis** – Positive, negative, neutral classification  
- **Medical Information Extraction** – Identifies symptoms, conditions, treatments  
- **Review Classification** – Classifies review content based on predefined labels  
- **Industrial Machinery Information** – Extracts technical data and operational context  
- **Fake News Detection** – Detects misinformation using evidence-based reasoning  

### 3. Performance Metrics

EcoRAG tracks:

- ⏱️ **Latency** – Inference time  
- ⚡ **Energy Consumption** – Joules consumed  
- 💾 **Memory Usage** – RAM utilization  
- 📈 **FLOPS per Token** – Efficiency metric  
- 🔁 **Throughput** – Tokens per second  
- 🧠 **Token Count** – Total tokens processed  

---

## 🚀 Features

- Modular architecture for easy extension  
- Batch processing for scalability  
- PostgreSQL and Neo4j integration  
- Graph-based reasoning (optional)  
- Works with local models (via Ollama)  
- Customizable prompt templates for each task  

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/WilliamBeckhauser/ecorag.git
cd ecorag
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Create a `.env` file**

```env
# Ollama LLM settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b

# PostgreSQL connection
POSTGRES_DBNAME=your_dbname
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Experiment schema/table
EXPERIMENT_SCHEMA=experiments
EXPERIMENT_TABLE=results
```

---

## ⚙️ Advanced Setup (Data Split + Graph RAG)

### 1. Environment Configuration (.env)

Create a `.env` file (or update the existing one) with the following variables:

```env
# API Tokens (set only if necessary)
OPENROUTER_TOKEN="if necessary"
API_KEY_OPENAI="if necessary"
API_KEY_GOOGLE="if necessary"
# Ollama does not require any token here

# PostgreSQL credentials (substitua pelos seus dados)
POSTGRES_DBNAME="postgres"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="postgres"
POSTGRES_HOST="localhost"
POSTGRES_PORT="5432"

# Neo4j credentials (substitua pelos seus dados)
NEO4J_URI=""
NEO4J_USER=""
NEO4J_PASSWORD=""
```

### 2. Data Preparation

After configuring the `.env` file, run the following command to clone the datasets (sst5, gossipcop, and politifact) and create schemas to separate the data into validation, test, original, experiments, and graphs:

```bash
python src/utils/data_split.py --z 2.576 --p 0.5 --e 0.02
```

### 3. Graph Creation

Next, create the graphs. The code currently uses the ollama3.1 8b model via Ollama. Customize the parameters as needed and run:

```bash
python src/utils/graph_creator.py \
  --neo4j_uri "your_uri" \
  --neo4j_user "neo4j" \
  --neo4j_password "your_passworw" \
  --postgres_schema "graph" \
  --postgres_table "tablename" \
  --id_column "id" \
  --title_column "title" \
  --classification_column "label"
```

### 4. Running the Graph RAG Experiment

After configuring or creating the graphs, run the following command to execute the experiment using graph data defined in the `.env` file. In this example, EcoRAG is used for fake news detection:

```bash
python ecorag.py \
  --source_schema test \
  --table_name politifact \
  --review_column title \
  --label_column label \
  --model_name llama3.1:8b \
  --provider ollama \
  --code_name ecorag \
  --execution_mode ecorag \
  --task fake_news
```

---

## 📁 Project Structure

```
ecorag/
├── src/
│   ├── config/
│   │   └── settings.py         # Loads env settings
│   ├── models/                 # (Reserved for future model code)
│   ├── utils/
│   │   ├── database.py         # PostgreSQL operations
│   │   ├── performance.py      # Metrics and system monitoring
│   │   ├── data_split.py       # Dataset loading and splitting
│   │   └── graph_creator.py    # Neo4j graph building
│   ├── providers/
│   │   └── llm_provider.py     # Interfaces with LLM APIs
│   ├── rag/
│   │   └── experiment.py       # Core experiment logic
│   └── main.py                 # CLI entry point for experiments
├── .env                        # Configuration file
├── ecorag.py                   # Main experiment runner
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```
