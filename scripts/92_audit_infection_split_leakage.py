
from pathlib import Path
from collections import defaultdict
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    ROOT / "data/raw/archive(4)/Infection Segmentation Data/"
    "Infection Segmentation Data"
)

OUTPUT = ROOT / "results/infection_mask_investigation"
OUTPUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["Train", "Val", "Test"]
CLASSES = ["COVID-19", "Non-COVID", "Normal"]

hash_records = defaultdict(list)

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


for split in SPLITS:
    for class_name in CLASSES:

        image_dir = DATASET / split / class_name / "images"

        for path in image_dir.glob("*"):

            if not path.is_file():
                continue

            file_hash = sha256_file(path)

            hash_records[file_hash].append({
                "split": split,
                "class": class_name,
                "filename": path.name,
                "path": str(path),
            })


records = []

for file_hash, samples in hash_records.items():

    split_names = sorted(set(
        sample["split"] for sample in samples
    ))

    if len(samples) > 1:

        for sample in samples:
            records.append({
                "sha256": file_hash,
                **sample,
                "duplicate_count": len(samples),
                "splits": ",".join(split_names),
                "cross_split_duplicate": len(split_names) > 1,
            })


df = pd.DataFrame(records)

print("\n========== IMAGE DUPLICATE AUDIT ==========\n")

print("Unique image hashes:", len(hash_records))
print("Duplicate image records:", len(df))

if not df.empty:
    cross_split = df[df["cross_split_duplicate"]]

    print(
        "Cross-split duplicate records:",
        len(cross_split)
    )

    print(
        "Cross-split duplicate groups:",
        cross_split["sha256"].nunique()
    )

    print("\nDuplicate distribution:")
    print(
        cross_split.groupby("splits")["sha256"]
        .nunique()
    )
else:
    print("No exact duplicate images found.")

output_path = OUTPUT / "infection_split_duplicate_audit.csv"

df.to_csv(output_path, index=False)

print("\nSaved:", output_path)

# to run this script, use the following command:
# python scripts/92_audit_infection_split_leakage.py