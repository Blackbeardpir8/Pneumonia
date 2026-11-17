
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

DATASET = (
    ROOT / "data/raw/archive(4)/Infection Segmentation Data/"
    "Infection Segmentation Data"
)

SAMPLES = [
    ("Train", "Non-COVID", "Non_COVID (28)"),
    ("Train", "Normal", "Normal (11766)"),
    ("Train", "COVID-19", "covid_1576"),
]

for split, class_name, filename in SAMPLES:

    base = DATASET / split / class_name

    print("\n" + "=" * 65)
    print(split, class_name, filename)
    print("=" * 65)

    for folder_name in [
        "images",
        "infection masks",
        "lung masks",
    ]:

        folder = base / folder_name

        matches = [
            p for p in folder.iterdir()
            if p.is_file() and p.stem == filename
        ]

        if len(matches) != 1:
            print(folder_name, "MATCH COUNT:", len(matches))
            continue

        path = matches[0]

        with Image.open(path) as img:
            array = np.array(img.convert("L"))

        print("\nFolder:", folder_name)
        print("Actual file:", path)
        print("Shape:", array.shape)
        print("Unique values:", np.unique(array)[:30])
        print("Foreground pixels:", np.count_nonzero(array > 0))
        print(
            "Foreground percentage:",
            round(np.mean(array > 0) * 100, 4)
        )

# to run this script, use the following command:
# python scripts/86_verify_suspicious_masks.py