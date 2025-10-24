import json, os, traceback, numpy as np, pandas as pd
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.quantum_info import Statevector, state_fidelity

GRPO_FILE = "./grpo_qiskit_generated_completions.json"
ORPO_FILE = "./orpo_qiskit_generated_completions.json"
OUTPUT_CSV = "./qiskit_benchmark_results.csv"

SV_BACKEND = Aer.get_backend("aer_simulator_statevector")
MC_BACKEND = Aer.get_backend("aer_simulator")

def safe_exec(code: str):
    """Safely execute generated code and return defined objects."""
    local_env = {}
    try:
        exec(code, {}, local_env)
        return local_env
    except Exception:
        return {}

def extract_circuit(local_env):
    """Return first QuantumCircuit in env if exists."""
    for val in local_env.values():
        if isinstance(val, QuantumCircuit):
            return val
    return None

def simulate_sv(qc):
    """Simulate and return statevector if possible."""
    try:
        tqc = transpile(qc, SV_BACKEND)
        result = SV_BACKEND.run(tqc).result()
        return True, Statevector(result.get_statevector(tqc))
    except Exception:
        return False, None

def simulate_counts(qc, shots=1024):
    try:
        if not any(inst.operation.name == "measure" for inst in qc.data):
            qc = qc.copy()
            qc.measure_all()
        tqc = transpile(qc, MC_BACKEND)
        result = MC_BACKEND.run(tqc, shots=shots).result()
        return True, result.get_counts()
    except Exception:
        return False, {}

def evaluate_entry(entry):
    """Compute compile success, sim success, fidelity, and depth."""
    result = {
        "task_id": entry.get("task_id"),
        "compile_success": 0,
        "sim_success": 0,
        "fidelity": np.nan,
        "depth": np.nan,
    }
    code = entry.get("generated_code", entry.get("completion", ""))

    env = safe_exec(code)
    qc = extract_circuit(env)
    if qc is None:
        return result

    result["compile_success"] = 1
    result["depth"] = qc.depth()

    ok_sv, sv = simulate_sv(qc)
    result["sim_success"] = int(ok_sv)

    ref_code = entry.get("reference") or entry.get("canonical_solution")
    if ref_code:
        ref_env = safe_exec(ref_code)
        ref_qc = extract_circuit(ref_env)
        if ref_qc:
            ok_ref, ref_sv = simulate_sv(ref_qc)
            if ok_ref and ok_sv:
                try:
                    fid = float(state_fidelity(ref_sv, sv))
                    result["fidelity"] = fid
                except Exception:
                    pass
    return result


def evaluate_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    results = []
    for d in data:
        try:
            results.append(evaluate_entry(d))
        except Exception as e:
            traceback.print_exc()
            continue
    return pd.DataFrame(results)


print("[INFO] Evaluating GRPO...")
df_grpo = evaluate_file(GRPO_FILE)
print("[INFO] Evaluating ORPO...")
df_orpo = evaluate_file(ORPO_FILE)

df_grpo["model"] = "GRPO"
df_orpo["model"] = "ORPO"
df = pd.concat([df_grpo, df_orpo], ignore_index=True)

summary = (
    df.groupby("model")
    .agg(
        compile_rate=("compile_success", "mean"),
        sim_rate=("sim_success", "mean"),
        mean_fidelity=("fidelity", "mean"),
        median_depth=("depth", "median"),
        count=("task_id", "count"),
    )
    .reset_index()
)

summary["compile_rate"] = summary["compile_rate"].round(3)
summary["sim_rate"] = summary["sim_rate"].round(3)
summary["mean_fidelity"] = summary["mean_fidelity"].round(3)
summary["median_depth"] = summary["median_depth"].round(2)

print("\n=== Benchmark Results ===")
print(summary.to_markdown(index=False))

summary.to_csv(OUTPUT_CSV, index=False)
print(f"\n[Saved results → {OUTPUT_CSV}]")
