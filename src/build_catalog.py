import pandas as pd
import re
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Building catalog dataset...")

# ----------------------------
# Load datasets
# ----------------------------
df = pd.read_csv(RAW_DIR / "animes.csv")
df3 = pd.read_csv(RAW_DIR / "animes2.csv")

# Clean column names
df.columns = df.columns.str.strip()
df3.columns = df3.columns.str.strip()

# ----------------------------
# Extract MAL ID from URL
# ----------------------------
def extract_mal_id(url):
    if pd.isna(url):
        return None
    match = re.search(r"/anime/(\d+)", str(url))
    return int(match.group(1)) if match else None

df["mal_id"] = df["mal_url"].apply(extract_mal_id)

# ----------------------------
# Align IDs
# ----------------------------
df.rename(columns={"animeID": "old_animeID"}, inplace=True)
df.rename(columns={"mal_id": "animeID"}, inplace=True)

# Create mapping table
id_map = df[["old_animeID", "animeID"]].dropna()

# Save mapping
id_map.to_csv(INTERIM_DIR / "anime_id_map.csv", index=False)

id_dict = dict(zip(id_map["old_animeID"], id_map["animeID"]))

print("ID mapping saved.")

# ----------------------------
# Build catalog dataset
# ----------------------------
df_subset = df[[
    "animeID",
    "sequel",
    "mal_url",
    "genres_detailed"
]].copy()

# Align dataset 3 key
df3.rename(columns={"id": "animeID"}, inplace=True)

# Merge datasets
df_final = df3.merge(df_subset, on="animeID", how="left")

# Save processed catalog
df_final.to_csv(PROCESSED_DIR / "anime_merged.csv", index=False)

print("Catalog dataset saved.")
print(df_final.info())