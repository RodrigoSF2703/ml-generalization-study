# Pipeline de Classificação de Síndromes Genéticas

Este repositório contém um pipeline completo de análise de embeddings extraídos de imagens de pacientes, visando a classificação de síndromes genéticas. O projeto realiza desde o pré-processamento dos dados até a geração de relatórios em PDF com gráficos e métricas de desempenho.

## Tecnologias Utilizadas

- Python 3.10+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- FPDF (geração de PDFs)
- TSNE (redução de dimensionalidade)
- K-Nearest Neighbors (KNN)

## Estrutura do Projeto
```
project/
│
├─ data/ # Dataset pickle fornecido
│ └─ mini_gm_public_v0.1.p
│
├─ outputs/ # Outputs gerados (gráficos e PDFs)
│
├─ src/
│ ├─ preprocess.py # Carregamento e preparação de dados
│ ├─ eda.py # Estatísticas e visualizações exploratórias
│ ├─ visualize.py # Funções para t-SNE e outros gráficos
│ ├─ classification.py # KNN e métricas de avaliação
│ └─ generate_report.py # Geração do PDF final
│
├─ main.py # Script principal que executa o pipeline completo
└─ requirements.txt # Dependências do projeto
```


## Objetivo do Projeto

O objetivo é analisar embeddings de imagens e classificar o syndrome_id de cada paciente utilizando um modelo KNN.

O pipeline realiza:

### Pré-processamento
- Carregamento e validação de embeddings
- Flatten da hierarquia em DataFrame

### Exploração de dados (EDA)
- Estatísticas de imagens por síndrome e por paciente
- Detecção de desequilíbrios de classes

### Visualização
- Redução de dimensionalidade via t-SNE
- Gráficos de clusters coloridos por síndrome

### Classificação KNN
- Avaliação usando distâncias Euclidiana e Cosseno
- Cross-validation 10-fold
- Cálculo de métricas: Accuracy, F1, Top-k Accuracy e AUC
- Determinação do valor ótimo de k

### Relatório Final
- PDF com metodologia, resultados, gráficos, tabela de métricas e insights

## Como Executar

### Preparação do Ambiente

Instale as dependências dentro do ambiente virtual:

```
pip install -r requirements.txt
```

### Rodar o Pipeline

Para gerar todos os resultados e o PDF final:

```
python main.py
```

Todos os gráficos e o PDF serão gerados na pasta `outputs/`.

## Conteúdo do PDF

O PDF final contém:

- **Metodologia**: Descrição do pré-processamento, algoritmo KNN e métricas
- **Resultados**: Gráficos de t-SNE e métricas (Accuracy, F1, Top-k, AUC)
- **Tabela de Métricas KNN**: Comparação entre distâncias Euclidiana e Cosseno de forma legível
- **Análise e Insights**: Interpretação dos resultados, separabilidade dos clusters e impacto do desequilíbrio de classes
- **Recomendações**: Sugestões de próximos passos e melhorias

## Saídas Esperadas

### Gráficos
- `tsne_plot.png`
- `accuracy_plot.png`
- `f1_plot.png`
- `topk_plot.png`
- `auc_plot.png`

### PDF Final
- `Relatorio_Classificacao_Sindromes.pdf`
- Tabela de métricas detalhada no PDF

## Próximos Passos / Melhorias Futuras

- Testar outros algoritmos (SVM, Random Forest, Redes Neurais)
- Aplicar redução de dimensionalidade (PCA) para acelerar KNN
- Usar técnicas de balanceamento de classes (SMOTE/oversampling)
- Ajustar top-k conforme necessidade clínica