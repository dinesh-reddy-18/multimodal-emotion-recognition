import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from pathlib import Path

# ============================================================
# Step 1: merge split assignment into text index (same fix as MFCC)
# Step 2: extract RoBERTa [CLS]-pooled embeddings for each fixed sentence
# ============================================================

text_path = Path("data/interim/ravdess_text_index.csv")
split_path = Path("data/interim/ravdess_split_index.csv")

text_df = pd.read_csv(text_path)
split_df = pd.read_csv(split_path)

split_lookup = split_df[["filename", "split"]].drop_duplicates(subset="filename")
text_df = text_df.drop(columns=["split"], errors="ignore")
text_df = text_df.merge(split_lookup, on="filename", how="left")

missing = text_df[text_df["split"].isna()]
if len(missing) > 0:
    print(f"WARNING: {len(missing)} rows have no split assigned!")
else:
    print("All rows matched to a split.")
print(text_df["split"].value_counts(dropna=False))

# ---- RoBERTa embedding extraction ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base").to(device)
model.eval()

# Only 2 unique sentences exist, so embed each once and reuse (fast + correct)
unique_texts = text_df["text"].unique()
embedding_cache = {}

with torch.no_grad():
    for sentence in unique_texts:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        # [CLS] token embedding (index 0) as the sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        embedding_cache[sentence] = cls_embedding
        print(f"Embedded: '{sentence}' -> shape {cls_embedding.shape}")

# Map embeddings back onto every row
embedding_dim = list(embedding_cache.values())[0].shape[0]
embedding_cols = [f"roberta_{i}" for i in range(embedding_dim)]

embeddings_matrix = text_df["text"].map(embedding_cache)
embeddings_df = pd.DataFrame(embeddings_matrix.tolist(), columns=embedding_cols, index=text_df.index)

final_df = pd.concat([text_df, embeddings_df], axis=1)

output_path = Path("data/interim/ravdess_text_features.csv")
final_df.to_csv(output_path, index=False)
print(f"\nSaved {len(final_df)} rows x {len(embedding_cols)} embedding dims to {output_path.resolve()}")
