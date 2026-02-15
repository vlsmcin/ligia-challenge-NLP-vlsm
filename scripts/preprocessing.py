import os
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def mark_caps(word: str) -> str:
	if word.isupper() and len(word) > 2:
		return word.lower() + "_ALLCAPS"
	return word


def clean_text(text: str) -> str:
	text = text.lower()
	text = re.sub(r"\d+", " NUM ", text)
	text = re.sub(r"([!?.,])", r" \1 ", text)
	tokens = word_tokenize(text)
	tokens = [mark_caps(token) for token in tokens]
	return " ".join(tokens)


def join_text(df: pd.DataFrame) -> pd.Series:
	return (df["clean_text"] + " " + df["clean_title"]).fillna("")


def main() -> None:
	base_dir = Path(__file__).resolve().parents[1]
	raw_dir = base_dir / "data" / "raw"
	processed_dir = base_dir / "data" / "processed"
	models_dir = base_dir / "models"

	train_path = raw_dir / "train.csv"
	final_test_path = raw_dir / "test.csv"

	print("[1/7] Carregando dados brutos...")
	train = pd.read_csv(train_path)
	final_test = pd.read_csv(final_test_path)
	print(f"Treino: {train.shape[0]} linhas | Teste final: {final_test.shape[0]} linhas")

	print("[2/7] Baixando recursos do NLTK...")
	nltk.download("punkt", quiet=True)
	nltk.download("punkt_tab", quiet=True)

	print("[3/7] Limpando textos...")
	train["clean_text"] = train["text"].apply(clean_text)
	train["clean_title"] = train["title"].apply(clean_text)
	final_test["clean_text"] = final_test["text"].apply(clean_text)
	final_test["clean_title"] = final_test["title"].apply(clean_text)

	train.drop(columns=["text", "title", "subject", "date"], inplace=True)
	final_test.drop(columns=["text", "title", "subject", "date"], inplace=True)

	vectorizer_word = TfidfVectorizer(
		analyzer="word",
		ngram_range=(1, 2),
		min_df=5,
		max_df=0.9,
		sublinear_tf=True,
	)

	vectorizer_char = TfidfVectorizer(
		analyzer="char",
		ngram_range=(3, 4),
		min_df=5,
		max_features=100000,
		dtype=np.float32,
	)

	print("[4/7] Dividindo conjuntos (treino/validacao/teste)...")
	X_train_set, X_test, y_train_set, y_test = train_test_split(
		train.drop(columns=["label"]),
		train["label"],
		test_size=0.1,
		random_state=42,
	)

	X_train, X_val, y_train, y_val = train_test_split(
		X_train_set,
		y_train_set,
		test_size=0.2,
		random_state=42,
	)

	train_text = join_text(X_train)
	val_text = join_text(X_val)
	test_text = join_text(X_test)
	final_test_text = join_text(final_test)

	print("[5/7] Vetorizando textos (TF-IDF word + char)...")
	X_word_train = vectorizer_word.fit_transform(train_text)
	X_char_train = vectorizer_char.fit_transform(train_text)

	models_dir.mkdir(parents=True, exist_ok=True)
	joblib.dump(vectorizer_word, models_dir / "vectorizer_word.joblib")
	joblib.dump(vectorizer_char, models_dir / "vectorizer_char.joblib")
	print("Vetorizadores salvos em models/")

	def vectorize(text: pd.Series):
		X_word = vectorizer_word.transform(text)
		X_char = vectorizer_char.transform(text)
		return hstack([X_word, X_char])

	X_train = hstack([X_word_train, X_char_train])
	X_val = vectorize(val_text)
	y_train = y_train.reset_index(drop=True)
	y_val = y_val.reset_index(drop=True)

	X_test = vectorize(test_text)
	y_test = y_test.reset_index(drop=True)

	X_final_test = vectorize(final_test_text)

	print("[6/7] Salvando matrizes e labels em data/processed...")
	processed_dir.mkdir(parents=True, exist_ok=True)
	save_npz(processed_dir / "X_train.npz", X_train)
	save_npz(processed_dir / "X_val.npz", X_val)
	save_npz(processed_dir / "X_test.npz", X_test)
	y_train.to_csv(processed_dir / "y_train.csv", index=False)
	y_val.to_csv(processed_dir / "y_val.csv", index=False)
	y_test.to_csv(processed_dir / "y_test.csv", index=False)
	save_npz(processed_dir / "X_final_test.npz", X_final_test)
	print("[7/7] Pre-processamento concluido.")


if __name__ == "__main__":
	main()
