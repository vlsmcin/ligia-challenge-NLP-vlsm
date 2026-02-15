# ligia-challenge-NLP-vlsm

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projeto de classificação de textos para detecção de desinformação digital. O objetivo é desenvolver e avaliar modelos de NLP capazes de distinguir notícias legítimas de conteúdos falsos com base em padrões linguísticos e semânticos. O projeto foi desenvolvido como parte da segunda etapa do processo seletivo da Liga Acadêmica de Inteligẽncia Artificial da UFPE.

## 📦 Pré-requisitos

- Python 3.12+
- Conda (Anaconda ou Miniconda)

---

## ⚙️ Setup do ambiente

### 1. Criar o ambiente virtual

```bash
conda create -n ligia-challenge-NLP python=3.12
```

### 2. Ativar o ambiente

```bash
conda activate ligia-challenge-NLP
```

### 3. Instalar as dependências

```bash
python -m pip install -r requirements.txt
```

## Project Organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── scripts            <- Scripts executaveis para pre-processamento e treino
│   ├── preprocessing.py <- Gera matrizes em data/processed e salva vetorizadores
│   └── training.py     <- Treina o modelo final e salva models/best_model.joblib
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
│
└── src   <- Source code for use in this project.
    │
    └── __init__.py             <- Makes src a Python module
```

## 🧪 Como executar

### 1. Gerar dados processados

```bash
python scripts/preprocessing.py
```

Ou usando make:

```bash
make preprocess
```

### 2. Treinar modelo final

```bash
python scripts/training.py
```

Ou usando make:

```bash
make train
```

O modelo treinado sera salvo em `models/best_model.joblib`.

--------

