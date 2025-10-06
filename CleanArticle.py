import pandas as pd
from pathlib import Path

import CorpusGenerator as cg

base = Path.cwd()
csv_path = base / "Articles.csv"

# --- 1) Load CSV and normalize ---
df = (pd.read_csv(csv_path)[["Content", "Summary"]].rename(columns={"Content": "article", "Summary": "summary"})
)

# --- 4) Basic cleanup ---
for col in ("article", "summary"):
    df[col] = (df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )

# Drop empty or NaN rows
df = df.dropna(subset=["article", "summary"])

# Remove any duplicate pairs
df = df.drop_duplicates(subset=["article", "summary"], keep="first").reset_index(drop=True)

print(df.info())
print(df.head(10))

all_text = " ".join(df["article"].astype(str).tolist())

corpus_dir = base / "corpus"
corpus_dir.mkdir(exist_ok=True)

tokens, token2idx, idx2token = cg.process_corpus(corpus_dir=str(corpus_dir), text=all_text)

# save
#out_csv = base / "final_news_summary.csv"
#df.to_csv(out_csv, index=False, encoding="utf-8")
