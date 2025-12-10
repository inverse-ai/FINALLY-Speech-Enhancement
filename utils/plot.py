import matplotlib.pyplot as plt
import re
import os
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, help="Path to validation log file")
parser.add_argument("--step", type=int, default=1000, help="Validation step interval")
args = parser.parse_args()

log_file = args.file
val_step = args.step

# Read file
with open(log_file, "r") as f:
    log_data = f.read()

# Extract values
utmos = [float(x) for x in re.findall(r"val_utmos:\s*([\d.]+)", log_data)]
wv_mos = [float(x) for x in re.findall(r"val_wvmos:\s*([\d.]+)", log_data)]
dnsmos = [float(x) for x in re.findall(r"val_dnsmos:\s*([\d.]+)", log_data)]
p808_mos = [float(x) for x in re.findall(r"val_P808_mos:\s*([\d.]+)", log_data)]
wb_pesq = [float(x) for x in re.findall(r"val_wb_pesq:\s*(-?[\d.]+)", log_data)]
stoi = [float(x) for x in re.findall(r"val_stoi:\s*([\d.]+)", log_data)]

# X-axis steps using validation step
steps = [val_step * (i + 1) for i in range(len(utmos))]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(steps, utmos, marker=".", label="utmos")
plt.plot(steps, wv_mos, marker=".", label="wvmos")
plt.plot(steps, dnsmos, marker=".", label="dnsmos")
plt.plot(steps, p808_mos, marker=".", label="p808_mos")
plt.plot(steps, wb_pesq, marker=".", label="wb-pesq")
plt.plot(steps, stoi, marker=".", label="stoi")

plt.xlabel("Step")
plt.ylabel("Score")
plt.title("Evaluation Metrics Over Steps")
plt.legend()
plt.grid(True, axis="y")

# Add vertical lines
for x in steps:
    if x % (2*val_step) == 0:
        plt.axvline(x=x, color="gray", linestyle="-", linewidth=1.0, alpha=0.4)
    else:
        plt.axvline(x=x, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

plt.tight_layout()

# Save plot under checkpoint directory
run_dir = os.path.dirname(log_file)
os.makedirs(run_dir, exist_ok=True)
plt.savefig(os.path.join(run_dir, "score.png"))