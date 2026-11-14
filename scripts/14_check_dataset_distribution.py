
from pathlib import Path
from collections import Counter

ROOT = Path("data/processed/segmentation")

for split in ["train", "val"]:
    image_dir = ROOT / split / "images"

    counts = Counter(
        path.name.split("_")[0]
        for path in image_dir.glob("*.png")
    )

    print(f"\n{split.upper()} DATASET DISTRIBUTION")
    print("-" * 40)

    total = sum(counts.values())

    for source, count in sorted(counts.items()):
        percentage = count / total * 100 if total else 0

        print(
            f"{source}: {count} images "
            f"({percentage:.2f}%)"
        )

    print(f"Total: {total}")

# to run 
