# src/visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE

def plot_tsne(df: pd.DataFrame, save_path: str = None, perplexity: int = 30, random_state: int = 42):
    """
    Reduz embeddings para 2D usando t-SNE e plota por síndrome.
    """
    print("🔹 Executando t-SNE (isso pode levar alguns segundos)...")
    embeddings = np.vstack(df["embedding"].values)  # corrigido

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state
    )
    reduced = tsne.fit_transform(embeddings)

    df_tsne = df.copy()
    df_tsne["tsne_x"] = reduced[:, 0]
    df_tsne["tsne_y"] = reduced[:, 1]

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x="tsne_x",
        y="tsne_y",
        hue="syndrome_id",
        palette="tab10",
        data=df_tsne,
        s=50,
        alpha=0.7
    )
    plt.title("t-SNE dos Embeddings por Síndrome")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📈 Gráfico t-SNE salvo em {save_path}")
        plt.close()
    else:
        plt.show()

    return df_tsne
