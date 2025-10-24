#!/bin/bash
set -e

echo "=============================================="
echo "🚀 QSPARK: End-to-End Quantum Code Fine-Tuning"
echo "=============================================="
echo

# Activate environment if needed
# source venv/bin/activate

# ---------- 1️⃣ Dataset Generation ----------
echo "🧩 Step 1: Generating dataset..."
python qiskit_to_orpo_pipeline.py

# ---------- 2️⃣ ORPO Fine-Tuning ----------
echo "⚙️ Step 2: Running ORPO fine-tuning..."
python orpo_training.py

# ---------- 3️⃣ GRPO Reinforcement Fine-Tuning ----------
echo "🤖 Step 3: Running GRPO fine-tuning..."
python grpo_qiskit_training.py

# ---------- 4️⃣ Benchmark Evaluation ----------
echo "📊 Step 4: Evaluating GRPO vs ORPO..."
python qiskit_benchmark_evaluation.py

# ---------- 5️⃣ Plot Visualization ----------
echo "🎨 Step 5: Generating plots..."
python qiskit_benchmark_plots.py

echo
echo "✅ All stages completed successfully!"
echo "----------------------------------------------"
echo "📁 Outputs:"
echo "  - Dataset: qiskit_humaneval_dataset.json"
echo "  - ORPO Model: ./orpo_outputs/lora_model/"
echo "  - GRPO Model: ./grpo_qiskit_outputs/lora_model/"
echo "  - Results: qiskit_benchmark_results.csv"
echo "  - Plots: ./benchmark_plots/"
echo
echo "🎉 QSPARK pipeline finished. Ready for analysis!"
