from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import SGDClassifier


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"

    print("[1/3] Carregando dados processados...")
    X_train = load_npz(data_dir / "X_train.npz")
    y_train = pd.read_csv(data_dir / "y_train.csv")["label"].values
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")

    print("[2/3] Treinando modelo com melhores parametros...")
    model = SGDClassifier(
        random_state=42,
        alpha=1e-5,
        class_weight=None,
        l1_ratio=0.85,
        loss="hinge",
        penalty="elasticnet",
    )
    model.fit(X_train, y_train)

    print("[3/3] Salvando modelo...")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "best_model.joblib")
    print("Treinamento concluido.")


if __name__ == "__main__":
    main()
