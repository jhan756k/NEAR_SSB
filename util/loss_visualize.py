import matplotlib.pyplot as plt
import re
import os

def plot(log_filepath):
    epochs = []
    train_losses = []
    val_losses = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_filepath = os.path.join(script_dir, log_filepath)

    epoch_pattern = r'epoch\s+(\d+)'
    train_loss_pattern = r'train\s+([0-9]*\.?[0-9]+)'
    val_loss_pattern = r'val\s+([0-9]*\.?[0-9]+)'

    with open(log_filepath, 'r') as file:
        for line in file:
            epoch_match = re.search(epoch_pattern, line)
            train_loss_match = re.search(train_loss_pattern, line)
            val_loss_match = re.search(val_loss_pattern, line)

            if epoch_match and train_loss_match and val_loss_match:
                epochs.append(int(epoch_match.group(1)))
                train_losses.append(float(train_loss_match.group(1)))
                val_losses.append(float(val_loss_match.group(1)))

    plt.figure(figsize=(12, 7))

    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('ma_loss_plot_v1.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    log_filepath = "ma_logv1.txt"
    plot(log_filepath)