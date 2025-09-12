
# Quantum LLM Fine-Tuning with ORPO & GRPO

## Abstract
Quantum circuits must be error-resilient, yet LLMs like Granite-20B-Code and StarCoder often output flawed Qiskit code.  
We fine-tuned a 32B model with two RL methods, **Group Relative Policy Optimization (GRPO)** and **Odds-Ratio Preference Optimization (ORPO)**, using a richly annotated synthetic dataset.

On the **Qiskit HumanEval** benchmark:
- ORPO reaches **56.29% Pass@1** (~+10pp over Granite-8B-QK).
- GRPO hits **49%**, both beating all general-purpose baselines.

On the **original HumanEval** benchmark:
- ORPO: **65.90%**
- GRPO: **63.00%**

**Task-level performance**:
- GRPO excels on *basic* tasks (42/54).
- ORPO dominates on *intermediate* tasks (41/68).
- Neither solves the *five advanced* tasks.

These results highlight clear gains but also remaining challenges in **AI-assisted quantum programming**.

---

## 🧱 Repository Structure
```
quantum-llm-orpo-grpo/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── data/
│   │   └── synthetic_dataset/      # scripts & format for synthetic data
│   ├── models/
│   │   ├── orpo.py                 # ORPO trainer
│   │   └── grpo.py                 # GRPO trainer
│   ├── evaluation/
│   │   ├── humaneval_qiskit.py     # Qiskit HumanEval runner
│   │   └── humaneval_original.py   # Original HumanEval runner
│   └── utils/
│       └── helpers.py              # common helpers & logging
└── scripts/
    ├── train_orpo.sh
    ├── train_grpo.sh
    └── evaluate.sh
```

## 🚀 Quickstart
```bash
git clone https://github.com/YOUR-USERNAME/quantum-llm-orpo-grpo.git
cd quantum-llm-orpo-grpo
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### Training
```bash
bash scripts/train_orpo.sh  # ORPO
bash scripts/train_grpo.sh  # GRPO
```

### Evaluation
```bash
bash scripts/evaluate.sh qiskit     # Qiskit HumanEval
bash scripts/evaluate.sh original   # Original HumanEval
```

## 📊 Reporting
Each run saves a JSON report under `runs/`:
```json
{
  "method": "ORPO",
  "benchmark": "qiskit_humaneval",
  "pass_at_1": 0.5629,
  "num_tasks": 167,
  "date": "2025-09-12T00:00:00Z",
  "model_name": "X"
}
```

## 📜 License
MIT License. See [LICENSE](LICENSE).
