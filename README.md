# Qiskit Dataset Generation & ORPO Training Pipeline
A comprehensive pipeline for generating Qiskit datasets and training language models using ORPO (Odds Ratio Preference Optimization).
## About this work:
This repository accompanies the paper “QSpark: Towards Reliable Qiskit Code Generation via ORPO and GRPO Optimization.”
The study introduces a unified framework that scrapes Qiskit-related repositories from GitHub, constructs HumanEval-style and preference datasets, and fine-tunes large language models using both ORPO and GRPO optimization techniques. The work demonstrates significant improvements in quantum code reliability, simulation success rates, and state fidelity across benchmark tasks, highlighting the importance of quantum-aware reward mechanisms for generative model training.

## Description

This repository contains two main components:

1. **`qiskit_to_orpo_pipeline.py`** - End-to-end script that scrapes Qiskit-related Python files from GitHub, extracts functions, and creates both HumanEval-format JSON datasets and ORPO training CSV files.

2. **`orpo_training.py`** - ORPO fine-tuning script using Unsloth for efficient training of language models on Qiskit code.

3. **`grpo_qiskit_training.py`** - Advanced GRPO (Group Relative Policy Optimization) training with quantum-specific reward functions for superior quantum code generation.

4. **`qiskit_benchmark_evaluation.py`** - Comprehensive evaluation script that compares GRPO vs ORPO model performance on quantum code generation tasks.

5. **`qiskit_benchmark_plots.py`** - Visualization script that generates publication-ready plots from benchmark results.

## Features

### Dataset Generation (`qiskit_to_orpo_pipeline.py`)
- 🔍 Search GitHub for Qiskit-related Python files
- ⚡ Concurrent downloading with configurable thread count
- 🛡️ Rate limiting to respect GitHub API limits
- 🧹 Automatic deduplication and difficulty analysis
- 📊 Dual output: HumanEval JSON + ORPO CSV formats
- 📈 Difficulty categorization (basic/intermediate/advanced)

### ORPO Training (`orpo_training.py`)
- 🚀 Efficient training with Unsloth framework
- 💾 LoRA fine-tuning for memory efficiency
- 📊 Wandb integration for experiment tracking
- 🎯 Optimized for Qiskit code generation tasks

### GRPO Training (`grpo_qiskit_training.py`)
- 🧠 Advanced Group Relative Policy Optimization
- ⚡ Multiple quantum-aware reward functions
- 🔬 AST + style conformance checking
- 🎯 Qiskit import/compile success validation
- 📊 Simulation fidelity vs. reference comparison
- 🏗️ Resource efficiency optimization
- 📝 XML-structured output formatting

### Benchmark Evaluation (`qiskit_benchmark_evaluation.py`)
- 📊 Comprehensive model comparison (GRPO vs ORPO)
- 🔬 Quantum circuit compilation success rates
- ⚡ Simulation success and fidelity metrics
- 📈 Circuit depth and resource efficiency analysis
- 📋 Automated CSV report generation
- 🎯 Side-by-side performance comparison

### Visualization (`qiskit_benchmark_plots.py`)
- 📊 Publication-ready benchmark plots
- 📈 Success rate comparisons (compile/simulation)
- 🔬 Fidelity distribution histograms
- 📊 Circuit depth box plots
- 🎨 Professional styling with seaborn/matplotlib
- 💾 High-resolution PNG outputs

## Requirements

### Basic Requirements
- Python 3.8+
- GitHub Personal Access Token
- Internet connection
- CUDA-compatible GPU (for training)

### Python Dependencies
- `requests>=2.25.1`
- `tqdm>=4.60.0`

### Training Dependencies (for ORPO/GRPO)
- `torch>=2.0.0`
- `unsloth>=2024.1`
- `transformers>=4.35.0`
- `datasets>=2.14.0`
- `trl>=0.7.0`
- `accelerate>=0.24.0`
- `qiskit>=0.45.0` (for GRPO quantum simulation)
- `qiskit-aer>=0.13.0` (for GRPO quantum simulation)
- `pandas>=1.5.0` (for evaluation metrics)
- `numpy>=1.21.0` (for numerical analysis)
- `matplotlib>=3.5.0` (for plotting)
- `seaborn>=0.11.0` (for statistical visualizations)

## Installation

1. Clone this repository
2. Install basic dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For ORPO/GRPO training, install additional dependencies:
   ```bash
   pip install unsloth trl transformers peft bitsandbytes accelerate datasets
   pip install qiskit qiskit-aer pandas numpy matplotlib seaborn  # for GRPO quantum simulation, evaluation, and visualization
   ```

## Usage

### 1. Dataset Generation

Generate Qiskit datasets from GitHub:

```bash
python qiskit_to_orpo_pipeline.py --token YOUR_GITHUB_TOKEN --max-files 100
```

**Arguments:**
- `--token` (required): GitHub personal access token
- `--max-files` (optional): Maximum number of files to download (default: 100)

**Output Files:**
- `qiskit_humaneval_dataset.json` - HumanEval format for evaluation
- `Formatted_ORPO_Dataset.csv` - ORPO training data

### 2. ORPO Training

Fine-tune a language model using the generated dataset:

```bash
python orpo_training.py
```

**Configuration (in script):**
- `MODEL_NAME`: Base model (default: "Qwen/Qwen2.5-Coder-32B-Instruct")
- `DATA_FILE`: Path to CSV dataset (default: "./Formatted_ORPO_Dataset.csv")
- `OUTPUT_DIR`: Output directory (default: "./orpo_outputs")
- `BATCH_SIZE`: Training batch size (default: 2)
- `EPOCHS`: Number of training epochs (default: 2)

### 3. GRPO Training (Advanced)

Advanced training with quantum-specific reward functions:

```bash
python grpo_qiskit_training.py
```

**Configuration (in script):**
- `MODEL_NAME`: Base model (set this before running)
- `DATA_PATH`: Path to JSON dataset (default: "./qiskit_humaneval_dataset.json")
- `OUTPUT_DIR`: Output directory (default: "./grpo_qiskit_outputs")
- `LORA_RANK`: LoRA rank (default: 16)
- `BATCH_SIZE`: Training batch size (default: 1)
- `EPOCHS`: Number of training epochs (default: 1)

**GRPO Reward Functions:**
- **Format**: XML structure + AST validation
- **Qiskit Import**: Valid import patterns (no deprecated APIs)
- **Compile & Simulate**: Circuit compilation and simulation success
- **Fidelity**: State fidelity comparison with reference
- **Resource Efficiency**: Circuit depth and gate count optimization

### 4. Model Evaluation

Compare GRPO vs ORPO model performance:

```bash
python qiskit_benchmark_evaluation.py
```

**Input Files:**
- `grpo_qiskit_generated_completions.json` - GRPO model outputs
- `orpo_qiskit_generated_completions.json` - ORPO model outputs

**Output:**
- `qiskit_benchmark_results.csv` - Detailed performance metrics
- Console summary with key statistics

**Evaluation Metrics:**
- **Compile Success Rate**: Percentage of syntactically valid code
- **Simulation Success Rate**: Percentage that successfully simulate
- **Fidelity**: Quantum state fidelity vs. reference solutions
- **Circuit Depth**: Resource efficiency comparison

### 5. Visualization

Generate publication-ready plots from benchmark results:

```bash
python qiskit_benchmark_plots.py
```

**Input Files:**
- `qiskit_benchmark_results.csv` - Summary statistics from evaluation
- `qiskit_benchmark_full_results.csv` - Detailed per-task results (optional)

**Output:**
- `benchmark_plots/success_rates.png` - Compile/simulation success rates
- `benchmark_plots/fidelity_distribution.png` - Fidelity distribution histogram
- `benchmark_plots/depth_boxplot.png` - Circuit depth comparison

**Plot Types:**
- **Success Rate Bar Charts**: Compile and simulation success rates
- **Fidelity Histograms**: Distribution of quantum state fidelity
- **Depth Box Plots**: Circuit depth resource efficiency comparison

## Getting a GitHub Token

1. Go to GitHub Settings > Developer settings > Personal access tokens
2. Generate a new token with appropriate permissions
3. Use the token with the `--token` argument

## Pipeline Workflow

### Complete Workflow

1. **Generate Dataset**: Run `qiskit_to_orpo_pipeline.py` to scrape GitHub and create training data
2. **Train Model**: Choose between:
   - **ORPO**: Run `orpo_training.py` for standard preference optimization
   - **GRPO**: Run `grpo_qiskit_training.py` for advanced quantum-specific training
3. **Generate Completions**: Use trained models to generate quantum code completions
4. **Evaluate**: Run `qiskit_benchmark_evaluation.py` to compare model performance
5. **Visualize**: Run `qiskit_benchmark_plots.py` to generate publication-ready plots

### Output Files

**Dataset Generation:**
- `qiskit_repos/` - Directory containing downloaded Python files
- `qiskit_humaneval_dataset.json` - HumanEval format for evaluation
- `Formatted_ORPO_Dataset.csv` - ORPO training data

**Training:**
- `orpo_outputs/lora_model/` - ORPO fine-tuned LoRA model
- `grpo_qiskit_outputs/lora_model/` - GRPO fine-tuned LoRA model
- `grpo_qiskit_merged/` - Optional merged GRPO model (if MERGE=True)
- Training logs and metrics (if using Wandb)

**Evaluation:**
- `qiskit_benchmark_results.csv` - Performance comparison metrics
- Console output with detailed statistics

**Visualization:**
- `benchmark_plots/success_rates.png` - Success rate bar charts
- `benchmark_plots/fidelity_distribution.png` - Fidelity distribution plots
- `benchmark_plots/depth_boxplot.png` - Circuit depth comparisons

## Contributing

Feel free to contribute to this project by submitting issues or pull requests.

## License

This project is open source and available under the [MIT License](LICENSE).
## Citation

If you use this repository or build upon this work, please cite:

```bibtex
@article{kheiri2025qspark,
  title={QSpark: Towards Reliable Qiskit Code Generation},
  author={Kheiri, Kiana and Aamir, Aamna and Miranskyy, Andriy and Ding, Chen},
  journal={arXiv preprint arXiv:2507.12642},
  year={2025}
}
```

