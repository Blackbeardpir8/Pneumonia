
from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "efficientnetb0_pneumonia_classifier_best.keras"
)

print("Loading classifier...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\n========== MODEL SUMMARY ==========")
model.summary(expand_nested=False)

print("\n========== TOP-LEVEL LAYERS ==========")

for index, layer in enumerate(model.layers):

    print(f"\nLayer {index}")
    print("Name:", layer.name)
    print("Type:", type(layer).__name__)

    try:
        print("Input shape:", layer.input_shape)
    except Exception:
        try:
            print("Input shape:", layer.input.shape)
        except Exception:
            print("Input shape: unavailable")

    try:
        print("Output shape:", layer.output.shape)
    except Exception:
        print("Output shape: unavailable")

    if isinstance(layer, tf.keras.Model):

        print("\nNested model layers:")

        for nested_index, nested_layer in enumerate(layer.layers):

            print(
                nested_index,
                nested_layer.name,
                type(nested_layer).__name__
            )

            try:
                print("  Output:", nested_layer.output.shape)
            except Exception:
                pass

print("\n========== EFFICIENTNET INSPECTION ==========")

backbone = model.get_layer("efficientnetb0")

print("Backbone input:", backbone.input.shape)
print("Backbone output:", backbone.output.shape)

target = backbone.get_layer("block5c_project_conv")

print("Grad-CAM target:", target.name)
print("Target output:", target.output.shape)

print("\nInspection completed.")


# to run this script, use the following command:
# python scripts/75a_inspect_classifier_architecture.py