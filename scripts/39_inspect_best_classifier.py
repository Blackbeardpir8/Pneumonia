
import sys
from pathlib import Path

import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiml.config import SAVED_MODELS_DIR

MODEL_PATH = (
    SAVED_MODELS_DIR
    / "efficientnetb0_pneumonia_classifier_best.keras"
)


def main():

    print("=" * 70)
    print("BEST CLASSIFIER ARCHITECTURE INSPECTION")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("\nModel:", model.name)
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

    print("\nTop-level layers:")

    for index, layer in enumerate(model.layers):

        print(
            index,
            layer.name,
            layer.__class__.__name__,
            "Trainable:", layer.trainable,
        )

    print("\nLooking for EfficientNet backbone...")

    backbones = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "efficientnet" in layer.name.lower()
    ]

    if not backbones:
        raise RuntimeError(
            "Nested EfficientNet backbone was not found."
        )

    for backbone in backbones:

        print("\nBackbone:", backbone.name)
        print("Total backbone layers:", len(backbone.layers))

        print("\nLast 50 backbone layers:")

        for index, layer in enumerate(
            backbone.layers[-50:],
            start=len(backbone.layers) - 50,
        ):

            print(
                index,
                layer.name,
                layer.__class__.__name__,
                "Trainable:", layer.trainable,
            )

        batch_norm_count = sum(
            isinstance(
                layer,
                tf.keras.layers.BatchNormalization,
            )
            for layer in backbone.layers
        )

        print("\nBatchNormalization layers:", batch_norm_count)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()


# to run this script, use the following command:
# python scripts/39_inspect_best_classifier.py