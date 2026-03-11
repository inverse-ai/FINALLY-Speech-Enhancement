import matplotlib.pyplot as plt
import numpy as np

def plot_training_losses(file_path, window_size=50):
    # The 6 specific losses we want to extract and plot
    plot_keys = [
        "total_loss_gen", 
        "wavlm_loss", 
        "stft_loss", 
        "gen_loss_ms-stft", 
        "feature_loss_ms-stft", 
        "total_loss_ms-stft"
    ]

    # Initialize data structures
    steps = []
    data = {key: [] for key in plot_keys}

    # 1. Parse the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line.startswith("Step"):
                continue
                
            try:
                # Split "Step X" from the rest of the losses
                step_part, losses_part = line.split(":", 1)
                step_num = int(step_part.replace("Step ", ""))
                
                loss_items = losses_part.split(",")
                
                current_losses = {}
                for item in loss_items:
                    k, v = item.strip().split("=")
                    current_losses[k] = float(v)
                
                steps.append(step_num)
                for key in plot_keys:
                    data[key].append(current_losses[key])
                    
            except Exception as e:
                continue

    # 2. Plotting
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 18), sharex=True)
    axes = axes.flatten()

    for i, key in enumerate(plot_keys):
        raw_loss = data[key]
        
        # Plot the raw, noisy data in the background (highly transparent)
        axes[i].plot(steps, raw_loss, color=f"C{i}", linewidth=1, alpha=0.2, label=f"{key} (Raw)")
        
        # Calculate and plot the 50-point running average
        if len(raw_loss) >= window_size:
            # np.convolve is a fast way to calculate moving averages
            weights = np.ones(window_size) / window_size
            smoothed_loss = np.convolve(raw_loss, weights, mode='valid')
            
            # mode='valid' drops the first (window_size - 1) incomplete points,
            # so we slice the steps array to match the new length
            smoothed_steps = steps[window_size - 1:]
            
            # Plot the smoothed line prominently
            axes[i].plot(smoothed_steps, smoothed_loss, color=f"C{i}", linewidth=2.5, label=f"{key} ({window_size}-pt Avg)")

        # Formatting
        axes[i].set_title(key, fontsize=12, fontweight='bold')
        axes[i].set_ylabel("Loss")
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend(loc="upper right")

    # Format the shared X-axis
    axes[-1].set_xlabel("Steps", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("loss_plots_smoothed.png", dpi=300)
    print("Plot saved as loss_plots_smoothed.png")

# Run the function
plot_training_losses('experiment/checkpoints/stage3_hifi_data/loss_log.txt', window_size=50)