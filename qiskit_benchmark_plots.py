import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULT_CSV = "./qiskit_benchmark_results.csv"
OUTPUT_DIR = "./benchmark_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(RESULT_CSV)
print(f"[INFO] Loaded benchmark summary:\n{df}\n")

# ----------------- PLOT 1 ---------------- #
# Compile / Simulation success rates
plt.figure(figsize=(6, 4))
rates = df.melt(
    id_vars="model",
    value_vars=["compile_rate", "sim_rate"],
    var_name="metric",
    value_name="rate",
)
sns.barplot(data=rates, x="metric", y="rate", hue="model", palette="viridis")
plt.title("Qiskit Compile & Simulation Success Rates")
plt.ylabel("Success Rate")
plt.xlabel("")
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/success_rates.png", dpi=300)
plt.close()

# ----------------- PLOT 2 ---------------- #
# Fidelity histogram 
try:
    df_full = pd.read_csv("./qiskit_benchmark_full_results.csv")
except FileNotFoundError:
    print("[WARN] Detailed fidelity CSV not found, using summary data only.")
else:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df_full, x="fidelity", hue="model", kde=True, bins=20, palette="plasma")
    plt.title("Fidelity Distribution (Qiskit-HumanEval)")
    plt.xlabel("State Fidelity")
    plt.ylabel("Count")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fidelity_distribution.png", dpi=300)
    plt.close()

# ----------------- PLOT 3 ---------------- #
# Depth comparison 
try:
    df_full = pd.read_csv("./qiskit_benchmark_full_results.csv")
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_full, x="model", y="depth", palette="viridis")
    plt.title("Circuit Depth Distribution")
    plt.ylabel("Depth")
    plt.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/depth_boxplot.png", dpi=300)
    plt.close()
except Exception as e:
    print("[WARN] Skipping depth plot:", e)

print(f"[DONE] Plots saved in {OUTPUT_DIR}/")
