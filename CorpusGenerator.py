import collections
import json
import os
import re

from tqdm import tqdm


def string_cleaner(text):
    """
    Clean and normalize raw text.

    - Lowercases all characters
    - Keeps only lowercase letters, spaces, and basic punctuation
    - Normalizes whitespace to a single space
    - Removes periods from common honorifics (mr., ms., mrs., dr.)
    """
    text = text.lower()
    text = re.sub(r"[^a-z,.?!':; ]", '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bmr\.', 'mr', text)
    text = re.sub(r'\bms\.', 'ms', text)
    text = re.sub(r'\bmrs\.', 'mrs', text)
    text = re.sub(r'\bdr\.', 'dr', text)
    return text


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
    token2idx['$'] = len(token2idx)
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token


def tokenize(text, token2idx):
    """
    Convert text into a list of token IDs.

    Uses greedy longest-match-first tokenization with a max token length of 5.
    """
    text = text.lower()
    text = re.sub(r"[^a-z,.?!':; $^]", '', text)
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
    tokenized_lines = [tokenize('^'+line+'$', token2idx) for line in tqdm(lines, desc="Tokenizing corpus")]
    tokenized_lines.sort(key=len)
    return tokenized_lines


def process_corpus(corpus_dir="corpus", text=None):
    """
    Build the cleaned corpus from scratch:
    - Reads Articles.csv and writes specified col to corpus.txt
    - Combines text from col to single txt file
    - Reads all .txt files in `corpus_dir`
    - Cleans text and splits into sentence-like lines
    - Builds vocabulary of top n-grams
    - Tokenizes lines
    - Saves results to JSON files
    """
    corpus_txt_path = os.path.join(corpus_dir, "corpus.txt")
    if text is not None:
        with open(corpus_txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    files = [f for f in os.listdir(corpus_dir) if f.endswith(".txt")]

    text = []
    for filename in tqdm(files, desc="Reading and cleaning files"):
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text.append(string_cleaner(f.read()))
    text = " ".join(text)

    # Split at end of sentence
    lines = re.sub(r'([?!.])', r'\1\n', text).splitlines()
    lines = [line.strip() for line in lines if 7 < len(line) < 250]

    # Build vocab and tokenize corpus
    token2idx, idx2token = identify_tokens(lines)
    tokens = tokenize_list(lines, token2idx)

    # Save everything
    print("Saving corpus and dictionaries...")
    with open("corpus.json", 'w', encoding="utf-8") as f:
        json.dump(tokens, f)
    with open("token2idx.json", 'w', encoding="utf-8") as f:
        json.dump(token2idx, f)
    with open("idx2token.json", 'w', encoding="utf-8") as f:
        json.dump(idx2token, f)

    return tokens, token2idx, idx2token


def get_cleaned_corpus():
    """
    Load processed corpus from disk if available, otherwise generate it.
    """
    if all(os.path.exists(f) for f in ("token2idx.json", "idx2token.json", "corpus.json")):
        with open("corpus.json", 'r', encoding="utf-8") as f:
            corpus = json.load(f)
        with open("token2idx.json", 'r', encoding="utf-8") as f:
            token2idx = json.load(f)
        with open("idx2token.json", 'r', encoding="utf-8") as f:
            idx2token = {int(idx): token for idx, token in json.load(f).items()}
        return corpus, token2idx, idx2token

    return process_corpus()


def get_dictionaries():
    """
    Load token dictionaries from disk if available, otherwise build them.
    """
    if all(os.path.exists(f) for f in ("token2idx.json", "idx2token.json")):
        with open("token2idx.json", 'r', encoding="utf-8") as f:
            token2idx = json.load(f)
        with open("idx2token.json", 'r', encoding="utf-8") as f:
            idx2token = {int(idx): token for idx, token in json.load(f).items()}
        return token2idx, idx2token

    return get_cleaned_corpus()[1:]

