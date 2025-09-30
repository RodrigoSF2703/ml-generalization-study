# src/classification.py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score, top_k_accuracy_score
from sklearn.preprocessing import LabelBinarizer


def run_knn_full_evaluation(df, max_k=15, cv_splits=10, top_k=3):
    """
    Roda KNN completo com cross-validation para diferentes valores de k.
    Calcula Accuracy, F1, Top-k Accuracy e AUC-ROC para distâncias Euclidiana e Cosseno.
    Retorna um DataFrame com todas as métricas.
    """
    print("🔹 Preparando dados para KNN...")
    X = np.vstack(df["embedding"].values)
    y = df["syndrome_id"].values

    # Cross-validation estratificada
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Armazenamento das métricas
    results = []

    for k in range(1, max_k + 1):
        acc_euclidean, acc_cosine = [], []
        f1_euclidean, f1_cosine = [], []
        topk_euclidean, topk_cosine = [], []
        auc_euclidean, auc_cosine = [], []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # ---------- Euclidean ----------
            knn_e = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            knn_e.fit(X_train, y_train)
            y_pred_e = knn_e.predict(X_test)
            y_pred_e_proba = knn_e.predict_proba(X_test)

            # Métricas
            acc_euclidean.append(knn_e.score(X_test, y_test))
            f1_euclidean.append(f1_score(y_test, y_pred_e, average="weighted"))
            topk_euclidean.append(top_k_accuracy_score(y_test, y_pred_e_proba, k=top_k))

            try:
                lb = LabelBinarizer().fit(y_train)
                y_test_bin = lb.transform(y_test)
                y_pred_bin = lb.transform(y_pred_e)
                auc_euclidean.append(roc_auc_score(y_test_bin, y_pred_bin, average="weighted", multi_class="ovr"))
            except:
                auc_euclidean.append(np.nan)

            # ---------- Cosine ----------
            knn_c = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            knn_c.fit(X_train, y_train)
            y_pred_c = knn_c.predict(X_test)
            y_pred_c_proba = knn_c.predict_proba(X_test)

            acc_cosine.append(knn_c.score(X_test, y_test))
            f1_cosine.append(f1_score(y_test, y_pred_c, average="weighted"))
            topk_cosine.append(top_k_accuracy_score(y_test, y_pred_c_proba, k=top_k))

            try:
                lb = LabelBinarizer().fit(y_train)
                y_test_bin = lb.transform(y_test)
                y_pred_bin = lb.transform(y_pred_c)
                auc_cosine.append(roc_auc_score(y_test_bin, y_pred_bin, average="weighted", multi_class="ovr"))
            except:
                auc_cosine.append(np.nan)

        results.append({
            "k": k,
            "euclidean_accuracy_mean": np.mean(acc_euclidean),
            "euclidean_accuracy_std": np.std(acc_euclidean),
            "euclidean_f1_mean": np.mean(f1_euclidean),
            "euclidean_f1_std": np.std(f1_euclidean),
            "euclidean_topk_mean": np.mean(topk_euclidean),
            "euclidean_topk_std": np.std(topk_euclidean),
            "euclidean_auc_mean": np.nanmean(auc_euclidean),
            "euclidean_auc_std": np.nanstd(auc_euclidean),
            "cosine_accuracy_mean": np.mean(acc_cosine),
            "cosine_accuracy_std": np.std(acc_cosine),
            "cosine_f1_mean": np.mean(f1_cosine),
            "cosine_f1_std": np.std(f1_cosine),
            "cosine_topk_mean": np.mean(topk_cosine),
            "cosine_topk_std": np.std(topk_cosine),
            "cosine_auc_mean": np.nanmean(auc_cosine),
            "cosine_auc_std": np.nanstd(auc_cosine),
        })

    df_results = pd.DataFrame(results)
    print("\n✅ Avaliação completa concluída!")
    return df_results


def plot_knn_results(df_results, metric="accuracy", save_path=None):
    """
    Plota desempenho do KNN em função de k para Euclidean e Cosine.
    metric: 'accuracy', 'f1', 'topk' ou 'auc'
    """
    plt.figure(figsize=(8, 6))
    plt.plot(df_results["k"], df_results[f"euclidean_{metric}_mean"], marker="o", label="Euclidean")
    plt.plot(df_results["k"], df_results[f"cosine_{metric}_mean"], marker="s", label="Cosine")

    plt.fill_between(df_results["k"],
                     df_results[f"euclidean_{metric}_mean"] - df_results[f"euclidean_{metric}_std"],
                     df_results[f"euclidean_{metric}_mean"] + df_results[f"euclidean_{metric}_std"],
                     alpha=0.2, color="blue")

    plt.fill_between(df_results["k"],
                     df_results[f"cosine_{metric}_mean"] - df_results[f"cosine_{metric}_std"],
                     df_results[f"cosine_{metric}_mean"] + df_results[f"cosine_{metric}_std"],
                     alpha=0.2, color="orange")

    plt.title(f"KNN Performance ({metric.upper()})")
    plt.xlabel("Número de vizinhos (k)")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📈 Gráfico {metric.upper()} salvo em {save_path}")
    else:
        plt.show()
