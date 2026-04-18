import pandas as pd
from pathlib import Path

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

print("Building interaction dataset...")

# ----------------------------
# Load mapping
# ----------------------------
id_map = pd.read_csv(INTERIM_DIR / "anime_id_map.csv")

id_dict = dict(zip(id_map["old_animeID"], id_map["animeID"]))

# ----------------------------
# Process ratings in chunks
# ----------------------------
ratings_path = RAW_DIR / "ratings.csv"
output_path = PROCESSED_DIR / "ratings_processed.csv"

chunksize = 2_000_000
first_chunk = True

for chunk in pd.read_csv(ratings_path, chunksize=chunksize):

    # Map IDs
    chunk["animeID"] = chunk["animeID"].map(id_dict)

    # Filter valid rows
    chunk = chunk[chunk["animeID"].notna()]

    # Optimize memory
    chunk["animeID"] = chunk["animeID"].astype("int32")
    chunk["userID"] = chunk["userID"].astype("int32")

    # Save incrementally
    chunk.to_csv(
        output_path,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

print("Ratings dataset processed.")