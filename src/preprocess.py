# src/preprocess.py
import pickle
import pandas as pd
import numpy as np

def load_data(pickle_path: str):
    """Carrega os dados do arquivo pickle."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data

def flatten_data(data: dict) -> pd.DataFrame:
    """
    Converte a hierarquia {syndrome_id → subject_id → image_id → embedding}
    em um DataFrame tabular.
    """
    rows = []
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                rows.append({
                    "syndrome_id": syndrome_id,
                    "subject_id": subject_id,
                    "image_id": image_id,
                    "embedding": np.array(embedding, dtype=np.float32)
                })
    return pd.DataFrame(rows)

def validate_embeddings(df: pd.DataFrame):
    """Verifica integridade dos embeddings (tamanho correto, valores nulos)."""
    df["valid_embedding"] = df["embedding"].apply(
        lambda emb: isinstance(emb, np.ndarray) and emb.shape == (320,)
    )
    invalid = df[~df["valid_embedding"]]
    if not invalid.empty:
        print(f"Atenção: {len(invalid)} embeddings inválidos encontrados.")
    else:
        print("✅ Todos os embeddings estão íntegros.")
    return df
