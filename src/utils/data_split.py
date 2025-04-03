import os
import pandas as pd
import requests
import zipfile
from io import BytesIO
import argparse
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


def download_fakenewsnet() -> tuple:
    """
    Downloads the FakeNewsNet repository from GitHub (if not already downloaded)
    and loads the CSV files for Politifact and GossipCop from the 'dataset' folder.

    Returns:
        tuple: (df_politifact, df_gossipcop)
    """
    DATASETS_DIR = "datasets"
    os.makedirs(DATASETS_DIR, exist_ok=True)
    repo_dir = os.path.join(DATASETS_DIR, "FakeNewsNet-master")
    if not os.path.exists(repo_dir):
        print("Downloading FakeNewsNet from GitHub...")
        url = "https://github.com/KaiDMML/FakeNewsNet/archive/refs/heads/master.zip"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(
                "Error downloading FakeNewsNet. Check your connection or URL.")
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(DATASETS_DIR)
        print("FakeNewsNet downloaded and extracted at:", repo_dir)
    else:
        print("FakeNewsNet is already downloaded at:", repo_dir)

    base_dataset_dir = os.path.join(repo_dir, "dataset")

    # Load Politifact CSV files
    politifact_fake_path = os.path.join(
        base_dataset_dir, "politifact_fake.csv")
    politifact_real_path = os.path.join(
        base_dataset_dir, "politifact_real.csv")

    df_pf_fake = pd.read_csv(politifact_fake_path)
    df_pf_real = pd.read_csv(politifact_real_path)
    df_pf_fake["label"] = "fake"
    df_pf_real["label"] = "real"
    df_politifact = pd.concat([df_pf_fake, df_pf_real], ignore_index=True)

    # Remove coluna "id" se existir e insere novo id sequencial
    if "id" in df_politifact.columns:
        df_politifact.drop("id", axis=1, inplace=True)
    df_politifact.reset_index(drop=True, inplace=True)
    df_politifact.insert(0, "id", df_politifact.index + 1)

    # Load GossipCop CSV files
    gossipcop_fake_path = os.path.join(base_dataset_dir, "gossipcop_fake.csv")
    gossipcop_real_path = os.path.join(base_dataset_dir, "gossipcop_real.csv")

    df_gc_fake = pd.read_csv(gossipcop_fake_path)
    df_gc_real = pd.read_csv(gossipcop_real_path)
    df_gc_fake["label"] = "fake"
    df_gc_real["label"] = "real"
    df_gossipcop = pd.concat([df_gc_fake, df_gc_real], ignore_index=True)

    # Remove coluna "id" se existir e insere novo id sequencial
    if "id" in df_gossipcop.columns:
        df_gossipcop.drop("id", axis=1, inplace=True)
    df_gossipcop.reset_index(drop=True, inplace=True)
    df_gossipcop.insert(0, "id", df_gossipcop.index + 1)

    print("Politifact shape:", df_politifact.shape)
    print("GossipCop shape:", df_gossipcop.shape)

    return df_politifact, df_gossipcop


def load_sst5() -> pd.DataFrame:
    """
    Downloads and loads the SST-5 dataset from Hugging Face (SetFit/sst5) and merges all splits.

    Returns:
        pd.DataFrame: A DataFrame containing all records from SST-5.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the 'datasets' package with: pip install datasets")

    print("Downloading SST-5 from Hugging Face...")
    dataset = load_dataset("SetFit/sst5")
    print("SST-5 downloaded.")

    df_list = []
    for split in dataset.keys():
        df_split = pd.DataFrame(dataset[split])
        print(f"  {split}: {df_split.shape[0]} records")
        df_list.append(df_split)

    df_sst5 = pd.concat(df_list, ignore_index=True)

    # Remove coluna "id" se existir e insere novo id sequencial
    if "id" in df_sst5.columns:
        df_sst5.drop("id", axis=1, inplace=True)
    df_sst5.reset_index(drop=True, inplace=True)
    df_sst5.insert(0, "id", df_sst5.index + 1)

    print("Total SST-5 records:", df_sst5.shape[0])
    return df_sst5


def create_schemas(engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS datasets"))
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS test"))
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS validator"))
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS graph"))
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS experiments"))
    print("Schemas 'datasets', 'test', 'validator' e 'graph' criados (caso não existam).")


def insert_datasets(engine, df_politifact: pd.DataFrame, df_gossipcop: pd.DataFrame, df_sst5: pd.DataFrame) -> None:
    """
    Insere cada dataset em uma tabela no schema 'datasets'.

    As tabelas são nomeadas: 'politifact', 'gossipcop' e 'sst5'.
    """
    df_politifact.to_sql("politifact", engine,
                         schema='datasets', if_exists='replace', index=False)
    df_gossipcop.to_sql("gossipcop", engine, schema='datasets',
                        if_exists='replace', index=False)
    df_sst5.to_sql("sst5", engine, schema='datasets',
                   if_exists='replace', index=False)
    print("Datasets inseridos no schema 'datasets'.")


def split_dataset(engine, table_name: str, Z: float, p: float, e: float) -> None:
    """
    Lê a tabela 'table_name' do schema 'datasets', calcula o tamanho da amostra com base nos parâmetros (Z, p, e)
    e divide os registros em três conjuntos:
      - Test: primeiros n registros da amostra final.
      - Validator: próximos 20% da amostra final.
      - Graph: registros restantes.

    Os subconjuntos são armazenados em tabelas com o mesmo nome nos schemas 'test', 'validator' e 'graph'.
    """
    query = f"SELECT * FROM datasets.{table_name}"
    df = pd.read_sql(query, engine)
    N = len(df)
    print(f"\nTabela '{table_name}' - Total de registros: {N}")

    n0 = (Z**2 * p * (1 - p)) / (e**2)
    n = n0 / (1 + (n0 / N))
    n = int(round(n))
    total_final = int(round(n * 1.2))

    if total_final > N:
        total_final = N
        n = int(round(total_final / 1.2))
        print("Ajuste: tamanho final da amostra limitado ao total do dataset.")

    print(f"Tamanho da amostra (n): {n}")
    print(f"Tamanho final da amostra (n + 20%): {total_final}")

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_sample = df_shuffled.iloc[:total_final].copy()
    test_df = final_sample.iloc[:n].copy()
    validator_df = final_sample.iloc[n:total_final].copy()
    graph_df = df_shuffled.iloc[total_final:].copy()

    print(f"Registros para Test: {len(test_df)}")
    print(f"Registros para Validator: {len(validator_df)}")
    print(f"Registros para Graph: {len(graph_df)}")

    test_df.to_sql(table_name, engine, schema='test',
                   if_exists='replace', index=False)
    validator_df.to_sql(table_name, engine, schema='validator',
                        if_exists='replace', index=False)
    graph_df.to_sql(table_name, engine, schema='graph',
                    if_exists='replace', index=False)

    print(f"Divisão e salvamento concluídos para a tabela '{table_name}'.")


def run_all_splits(Z: float, p: float, e: float) -> None:
    """
    Carrega todos os datasets, cria os schemas necessários, insere os datasets no schema 'datasets'
    e divide cada tabela em conjuntos de test, validator e graph.

    Args:
        Z (float): Valor Z para o nível de confiança.
        p (float): Proporção estimada.
        e (float): Margem de erro.
    """
    print("Carregando datasets...")
    df_politifact, df_gossipcop = download_fakenewsnet()
    df_sst5 = load_sst5()

    dbname = os.getenv("POSTGRES_DBNAME")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")

    engine = create_engine(
        f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

    create_schemas(engine)
    insert_datasets(engine, df_politifact, df_gossipcop, df_sst5)

    for table in ["politifact", "gossipcop", "sst5"]:
        split_dataset(engine, table, Z, p, e)

    print("\nProcessamento concluído para todas as tabelas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Divide os datasets no schema 'datasets' em conjuntos de test, validator e graph."
    )
    parser.add_argument("--z", type=float, default=2.576,
                        help="Valor Z para o nível de confiança (default: 2.576 para 99%% de confiança)")
    parser.add_argument("--p", type=float, default=0.5,
                        help="Proporção estimada (default: 0.5)")
    parser.add_argument("--e", type=float, default=0.02,
                        help="Margem de erro (default: 0.02)")
    args = parser.parse_args()

    run_all_splits(args.z, args.p, args.e)


# python src/utils/data_split.py --z 2.576 --p 0.5 --e 0.02
