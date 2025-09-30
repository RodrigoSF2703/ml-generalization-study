from fpdf import FPDF
import os
import pandas as pd


class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Relatório Técnico - Classificação de Síndromes", 0, 1, "C")
        self.ln(5)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 14)
        self.set_text_color(0, 0, 128)
        self.cell(0, 10, title, 0, 1)
        self.ln(2)

    def chapter_body(self, body):
        self.set_font("Arial", "", 12)
        self.set_text_color(0, 0, 0)
        body = body.replace("–", "-").replace("…", "...")
        self.multi_cell(0, 6, body)
        self.ln(4)

    def add_image(self, image_path, width=180):
        if os.path.exists(image_path):
            self.image(image_path, w=width)
            self.ln(5)

    def add_knn_table(self, df: pd.DataFrame):
        """
        Cria uma tabela legível para as métricas KNN, comparando Euclidean x Cosine
        para cada valor de k e cada métrica (accuracy, f1, topk, auc)
        """
        self.chapter_title("3. Tabela de Métricas KNN")

        # Configuração da tabela
        self.set_font("Arial", "B", 10)
        col_widths = [10, 25, 25, 25]  # k, Métrica, Euclidean, Cosine
        headers = ["k", "Métrica", "Euclidean", "Cosine"]
        th = 6
        for h, w in zip(headers, col_widths):
            self.cell(w, th, h, 1, 0, "C")
        self.ln(th)

        # Linhas da tabela
        self.set_font("Arial", "", 9)
        metrics = ["accuracy", "f1", "topk", "auc"]
        for idx, row in df.iterrows():
            k = int(row["k"])
            for m in metrics:
                self.cell(col_widths[0], th, str(k), 1, 0, "C")
                self.cell(col_widths[1], th, m.upper(), 1, 0, "C")
                eu_val = row.get(f"euclidean_{m}_mean", 0)
                co_val = row.get(f"cosine_{m}_mean", 0)
                self.cell(col_widths[2], th, f"{eu_val:.3f}", 1, 0, "C")
                self.cell(col_widths[3], th, f"{co_val:.3f}", 1, 0, "C")
                self.ln(th)
        self.ln(4)


def generate_pdf_report(pdf_path, graphs_paths, knn_metrics: pd.DataFrame = None):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Metodologia
    pdf.chapter_title("1. Metodologia")
    pdf.chapter_body(
        "Carregamos os dados de embeddings de imagens de pacientes, convertendo a hierarquia em DataFrame, validando embeddings, "
        "aplicando t-SNE para visualização e utilizando KNN (Euclidean e Cosine) para classificação com métricas: Accuracy, F1, Top-3 e AUC.\n"
        "Cross-validation estratificada (10-fold) foi aplicada para garantir avaliação robusta e balanceamento das classes."
    )

    # 2. Resultados (gráficos)
    pdf.chapter_title("2. Resultados")
    pdf.chapter_body(
        "Gráficos abaixo mostram a performance do KNN considerando diferentes métricas e tipos de distância."
    )
    for graph in graphs_paths:
        pdf.add_image(graph, width=180)

    # 3. Tabela de métricas KNN
    if knn_metrics is not None:
        pdf.add_knn_table(knn_metrics)

    # 4. Análise e Insights
    pdf.chapter_title("4. Análise e Insights")
    pdf.chapter_body(
        "- Cosine Distance apresentou melhor separabilidade e acurácia.\n"
        "- Valor ótimo de k entre 7-8, balanceando bias e variance.\n"
        "- Clusters no t-SNE confirmam que embeddings contêm informações discriminativas.\n"
        "- Top-3 Accuracy indica que previsões próximas da classe correta são frequentes mesmo quando há erro.\n"
        "- Desbalanceamento de classes impacta F1 e AUC, mas o modelo mantém boa performance geral."
    )

    # 5. Recomendações
    pdf.chapter_title("5. Recomendações")
    pdf.chapter_body(
        "- Testar outros algoritmos (SVM, Random Forest, Redes Neurais).\n"
        "- Aplicar PCA para redução de dimensionalidade e acelerar KNN.\n"
        "- Considerar técnicas de balanceamento de classes (SMOTE/oversampling).\n"
        "- Ajustar top-k conforme necessidade clínica."
    )

    pdf.output(pdf_path)
    print(f"\n✅ PDF gerado em: {pdf_path}")
