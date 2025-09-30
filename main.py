import os
from src.preprocess import load_data, flatten_data, validate_embeddings
from src.eda import dataset_stats, plot_images_per_syndrome, plot_images_per_subject
from src.visualize import plot_tsne
from src.classification import plot_knn_results, run_knn_full_evaluation
from src.generate_report import generate_pdf_report

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    pickle_path = "data/mini_gm_public_v0.1.p"

    # ---------------- Carregamento e pré-processamento ----------------
    print("🔹 Carregando dados...")
    data = load_data(pickle_path)

    print("🔹 Convertendo hierarquia em DataFrame...")
    df = flatten_data(data)

    print("🔹 Validando embeddings...")
    df = validate_embeddings(df)

    print("\n✅ Dataset processado com sucesso!")
    dataset_stats(df)

    # ---------------- EDA ----------------
    plot_images_per_syndrome(df)
    plot_images_per_subject(df)

    # ---------------- t-SNE ----------------
    tsne_path = os.path.join(OUTPUT_DIR, "tsne_plot.png")
    df_tsne = plot_tsne(df, save_path=tsne_path)

    # ---------------- Classificação KNN ----------------
    print("\n🔹 Rodando classificação KNN completa...")
    df_results = run_knn_full_evaluation(df)

    graph_paths = []
    for metric in ["accuracy", "f1", "topk", "auc"]:
        print(f"\n🔹 Gerando gráfico: {metric.upper()}")
        graph_path = os.path.join(OUTPUT_DIR, f"{metric}_plot.png")
        plot_knn_results(df_results, metric=metric, save_path=graph_path)
        graph_paths.append(graph_path)

    # ---------------- Gera PDF final completo ----------------
    pdf_path = os.path.join(OUTPUT_DIR, "Relatorio_Classificacao_Sindromes.pdf")
    generate_pdf_report(pdf_path, [tsne_path] + graph_paths, knn_metrics=df_results)

    print("\n✅ Pipeline concluído! PDF final gerado com gráficos e tabela de métricas legível.")

if __name__ == "__main__":
    main()
