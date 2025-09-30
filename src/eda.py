# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_stats(df: pd.DataFrame):
    """Mostra estatísticas básicas do dataset."""
    print("\n📊 Estatísticas do Dataset:")
    print(f"Total de imagens: {len(df)}")
    print(f"Número de síndromes: {df['syndrome_id'].nunique()}")
    print(f"Número de sujeitos: {df['subject_id'].nunique()}")
    print("\nImagens por síndrome:")
    print(df["syndrome_id"].value_counts())

def plot_images_per_syndrome(df: pd.DataFrame, save_path: str = None):
    """Plota distribuição de imagens por síndrome."""
    plt.figure(figsize=(10, 5))
    sns.countplot(x="syndrome_id", data=df, order=df["syndrome_id"].value_counts().index)
    plt.title("Distribuição de Imagens por Síndrome")
    plt.xlabel("Síndrome")
    plt.ylabel("Número de Imagens")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📈 Gráfico salvo em {save_path}")
    else:
        plt.show()

def plot_images_per_subject(df: pd.DataFrame, save_path: str = None):
    """Plota distribuição de imagens por sujeito (histograma)."""
    counts = df.groupby("subject_id").size()

    plt.figure(figsize=(8, 5))
    sns.histplot(counts, bins=20, kde=False)
    plt.title("Distribuição de Imagens por Sujeito")
    plt.xlabel("Número de Imagens por Sujeito")
    plt.ylabel("Frequência")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📈 Gráfico salvo em {save_path}")
    else:
        plt.show()
