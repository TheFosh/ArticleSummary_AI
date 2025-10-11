import collections
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from pandas import read_csv
from torch.utils.data import Dataset
from tqdm import tqdm


def output_clean_data():
    csv_path = "data.csv"

    # --- 1) Load CSV and normalize ---
    df = (pd.read_csv(csv_path, na_values=['nan', 'NaN'])[["Content", "Summary"]].rename(columns={"Content": "article", "Summary": "summary"})
    )

    # --- 4) Basic cleanup ---
    for col in ("article", "summary"):
        df[col] = (df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        )

    # Remove any duplicate pairs
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    # Drop empty or NaN rows
    df.replace(['', 'nan', 'NULL', 'None', 'N/A'], np.nan, inplace=True)
    df.dropna(inplace = True)

    print(df.info())
    print(df.head(10))
    print(df.isnull().values.any())

    # save
    out_csv = "final_news_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print(df.isnull().values.any())

def process_corpus(corpus_dir="final_news_summary.csv"):
    """
    Build the cleaned corpus from scratch:

    - Reads all .txt files in `corpus_dir`
    - Cleans text and splits into sentence-like lines
    - Builds vocabulary of top n-grams
    - Tokenizes lines
    - Saves results to JSON files
    """
    df = pd.read_csv(corpus_dir)
    LIMITER = 50000
    # Select variables
    y = df.iloc[: LIMITER, -1].copy().to_numpy()
    X = df.iloc[: LIMITER, 0].copy().to_numpy()

    # Build vocab and tokenize corpus
    a_token2idx, a_idx2token = identify_tokens(X)
    t_token2idx, t_idx2token = identify_tokens(y)
    a_tokens = tokenize_list(X, a_token2idx)
    t_tokens = tokenize_list(y, t_token2idx)

    a_tokens, t_tokens = sort_dataset(a_tokens, t_tokens)

    # Save everything
    print("Saving corpus and dictionaries...")
    with open("corpus.json", 'w', encoding="utf-8") as f:
        json.dump(a_tokens, f)
    with open("token2idx.json", 'w', encoding="utf-8") as f:
        json.dump(a_token2idx, f)
    with open("idx2token.json", 'w', encoding="utf-8") as f:
        json.dump(a_idx2token, f)
    with open("t_corpus.json", 'w', encoding="utf-8") as f:
        json.dump(t_tokens, f)
    with open("t_token2idx.json", 'w', encoding="utf-8") as f:
        json.dump(t_token2idx, f)
    with open("t_idx2token.json", 'w', encoding="utf-8") as f:
        json.dump(t_idx2token, f)

    return a_tokens, a_token2idx, a_idx2token, t_token2idx, t_idx2token

def sort_dataset(input_lines, target_lines):
    """
    Sort input-output pairs together by input length descending,
    preserving correspondence between input and output.
    """
    paired = list(zip(input_lines, target_lines))
    paired.sort(key=lambda x: len(x[0]), reverse=True)  # descending length
    sorted_inputs, sorted_targets = zip(*paired)
    return list(sorted_inputs), list(sorted_targets)

def identify_tokens(lines, num_ngrams=2500):
    """
    Build a vocabulary of the most frequent character n-grams.

    - Counts all n-grams from length 1 to 5
    - Keeps only the top `num_ngrams`
    - Returns dictionaries for both token->id and id->token
    """
    ngram_counts = collections.Counter()

    for line in tqdm(lines, desc=f"Creating and indexing top {num_ngrams} n-grams"):
        length = len(line)
        for n in range(1, 6):  # n-grams of length 1 to 5
            for i in range(length - n + 1):
                ngram_counts[line[i:i + n]] += 1

    top_n = [token for token, _ in ngram_counts.most_common(num_ngrams - 2)]
    token2idx = {token: idx for idx, token in enumerate(top_n)}
    token2idx['^'] = len(token2idx)
    token2idx['~'] = len(token2idx)
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token


def tokenize(text, token2idx):
    """
    Convert text into a list of token IDs.

    Uses greedy longest-match-first tokenization with a max token length of 5.
    """
    text = text.lower()
    text = re.sub(r"[^a-z,.?!':; ~^]", '', text)
    tokens = []
    idx = 0
    while idx < len(text):
        for token_length in range(min(5, len(text) - idx), 0, -1):
            substring = text[idx: idx + token_length]
            if substring in token2idx:
                tokens.append(token2idx[substring])
                idx += token_length
                break
        else:
            # If no token matched (shouldn't usually happen), skip one char
            idx += 1
    return tokens


def tokenize_list(lines, token2idx):
    """
    Tokenize a list of lines and return them sorted by length.
    """
    tokenized_lines = [tokenize('^'+line+'~', token2idx) for line in tqdm(lines, desc="Tokenizing corpus")]

    return tokenized_lines


def get_cleaned_corpus():
    """
    Load processed corpus from disk if available, otherwise generate it.
    """
    if all(os.path.exists(f) for f in ("t_idx2token.json","t_token2idx.json", "token2idx.json", "idx2token.json", "corpus.json", "t_corpus.json")):
        with open("corpus.json", 'r', encoding="utf-8") as f:
            corpus = json.load(f)
        with open("token2idx.json", 'r', encoding="utf-8") as f:
            token2idx = json.load(f)
        with open("idx2token.json", 'r', encoding="utf-8") as f:
            idx2token = {int(idx): token for idx, token in json.load(f).items()}
        with open("t_corpus.json", 'r', encoding="utf-8") as f:
            t_corpus = json.load(f)
        with open("t_token2idx.json", 'r', encoding="utf-8") as f:
            t_token2idx = json.load(f)
        with open("t_idx2token.json", 'r', encoding="utf-8") as f:
            t_idx2token = {int(idx): token for idx, token in json.load(f).items()}
        return corpus, token2idx, idx2token, t_corpus, t_token2idx, t_idx2token

    return process_corpus()

def get_cleaned_out(corpus_dir = "final_news_summary.csv"):
    df = pd.read_csv(corpus_dir)

    LIMITER = 10000
    # Select variables
    y = df.iloc[:LIMITER, -1].copy().to_numpy()
    X = df.iloc[:LIMITER, 0].copy().to_numpy()

    print(len(X))

    # Build vocab and tokenize corpus
    t_token2idx, t_idx2token = identify_tokens(y)
    t_tokens = tokenize_list(y, t_token2idx)

    with open("t_corpus.json", 'w', encoding="utf-8") as f:
        json.dump(t_tokens, f)

def get_only_corpus():
    with open("corpus.json", 'r', encoding="utf-8") as f:
        corpus = json.load(f)
    with open("t_corpus.json", 'r', encoding="utf-8") as f:
        t_corpus = json.load(f)

    return corpus, t_corpus


class ArticleData(Dataset):
    def __init__(self, max_length = 5):
        self.corpus, self.a_token2idx, self.a_idx2token, self.t_corpus, self.t_token2idx, self.t_idx2token = get_cleaned_corpus()
        self.max_length = max_length

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sentence = self.corpus[idx]
        target = self.t_corpus[idx]
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class TestArticleData(Dataset):
    def __init__(self, max_length = 5):
        self.corpus, self.t_corpus = get_only_corpus()

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sentence = self.corpus[idx]
        target = self.t_corpus[idx]
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# process_corpus()