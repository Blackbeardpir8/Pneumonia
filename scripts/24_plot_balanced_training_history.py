
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

history_path = Path(
    "results/segmentation/balanced_training_history.csv"
)

output_dir = Path("results/segmentation")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(history_path)

# Convert zero-based epoch numbers to human-readable epochs
epochs = df["epoch"] + 1

# Dice curves
plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    df["dice_coefficient"],
    label="Training Dice"
)

plt.plot(
    epochs,
    df["val_dice_coefficient"],
    label="Validation Dice"
)

best_index = df["val_dice_coefficient"].idxmax()

plt.scatter(
    epochs.iloc[best_index],
    df.loc[best_index, "val_dice_coefficient"],
    marker="*",
    s=180,
    label="Best validation epoch"
)

plt.xlabel("Epoch")
plt.ylabel("Dice Coefficient")
plt.title("Balanced U-Net - Dice History")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "balanced_dice_history.png",
    dpi=150
)

plt.close()

# Loss curves
plt.figure(figsize=(9, 5))

plt.plot(
    epochs,
    df["loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    df["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy Loss")
plt.title("Balanced U-Net - Loss History")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    output_dir / "balanced_loss_history.png",
    dpi=150
)

plt.close()

print("Training curves saved:")
print(output_dir / "balanced_dice_history.png")
print(output_dir / "balanced_loss_history.png")


# to run script, use the following command:
# python scripts/24_plot_balanced_training_history.py
