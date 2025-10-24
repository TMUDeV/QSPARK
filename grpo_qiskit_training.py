import os, re, json, math, torch, ast, warnings
from typing import List, Dict, Any, Tuple

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct" 
DATA_PATH  = "./qiskit_humaneval_dataset.json"  
OUTPUT_DIR = "./grpo_qiskit_outputs"
MERGED_DIR = "./grpo_qiskit_merged"            
LORA_RANK  = 16
MAX_SEQ    = 4096
BATCH_SIZE = 1
GRAD_ACCUM = 2
EPOCHS     = 1
LR         = 5e-6
NUM_GENERATIONS = 2
MAX_PROMPT_LEN   = 512
MAX_COMPLETION_LEN = 256
USE_WANDB = False  

warnings.filterwarnings("ignore", category=UserWarning)

from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchDPOTrainer
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import Aer

SV_BACKEND = Aer.get_backend("aer_simulator_statevector")
MC_BACKEND = Aer.get_backend("aer_simulator")  


def load_qiskit_dataset(path:str):
    ds = load_dataset("json", data_files=path, split="train")
    SYSTEM_PROMPT = (
        "Return XML blocks strictly:\n"
        "<reasoning>...</reasoning>\n<answer>...</answer>\n"
        "Where <answer> contains ONLY a valid Python function definition."
    )
    def _map(ex):
        msg = [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": ex["prompt"]}
        ]
        ref = ex.get("canonical_solution","").strip()
        return {"prompt":msg, "reference":ref}
    return ds.map(_map, remove_columns=[c for c in ds.column_names if c not in []])

dataset = load_qiskit_dataset(DATA_PATH)

dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ,
    dtype = dtype,
    load_in_4bit = True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_RANK,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

XML_SOFT = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)

def extract_answer(xml_text:str)->str:
    try:
        return xml_text.split("<answer>")[1].split("</answer>")[0].strip()
    except Exception:
        return ""

def safe_parse_func(src:str)->Tuple[bool,str]:
    """Sanity: single def, compilable, no obvious shell/IO."""
    try:
        t = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax_error: {e}"
    fdefs = [n for n in ast.walk(t) if isinstance(n, ast.FunctionDef)]
    if len(fdefs) != 1:
        return False, "must_contain_exactly_one_function_def"
    bad = ("os.system", "subprocess", "sys.exit", "open(", "requests.", "socket.")
    if any(b in src for b in bad):
        return False, "banned_calls"
    return True, "ok"

def is_qiskit_import_ok(src:str)->Tuple[bool,str]:
    if "import qiskit" in src or "from qiskit" in src:
        if "from qiskit import Aer" in src or "qiskit.Aer" in src:
            return False, "deprecated_aer_import"
        return True, "qiskit_import_ok"
    return False, "missing_qiskit_import"

def build_callable(src:str):
    """Exec the function into a dict and return the callable."""
    names = {}
    try:
        exec(src, {}, names)
    except Exception as e:
        return None, f"exec_error: {e}"
    fn = next((v for v in names.values() if callable(v)), None)
    if not fn:
        return None, "no_callable_found"
    return fn, "ok"

def reference_adapter(ref_src:str):
    """From canonical solution → a callable for checking behavior."""
    return build_callable(ref_src)

def try_make_circuit(fn) -> Tuple[bool, Any]:
    """Attempt to call the function in a conservative way to get a QuantumCircuit."""
    try:
        try:
            qc = fn(2)  
        except TypeError:
            qc = fn()
        if not isinstance(qc, QuantumCircuit):
            return False, "not_a_circuit"
        return True, qc
    except Exception as e:
        return False, f"fn_call_error: {e}"

def simulate_statevector(qc:QuantumCircuit) -> Tuple[bool, Any]:
    try:
        tqc = transpile(qc, SV_BACKEND)
        res = SV_BACKEND.run(tqc).result()
        sv = Statevector(res.get_statevector(tqc))
        return True, sv
    except Exception as e:
        return False, f"sv_sim_error: {e}"

def simulate_counts(qc:QuantumCircuit, shots:int=2048) -> Tuple[bool, Dict[str,int]]:
    try:
        mqc = qc.copy()
        if not any(inst.operation.name == "measure" for inst in mqc.data):
            mqc.measure_all()
        tqc = transpile(mqc, MC_BACKEND)
        res = MC_BACKEND.run(tqc, shots=shots).result()
        return True, res.get_counts()
    except Exception as e:
        return False, f"counts_sim_error: {e}"

def r_format(prompts, completions, **_):
    """Enforce XML container + single function in <answer> block."""
    scores = []
    for c in completions:
        text = c[0]["content"]
        ok_xml = 1.0 if XML_SOFT.search(text) else 0.0
        ans = extract_answer(text)
        ok_ast, _ = safe_parse_func(ans)
        scores.append(0.5*ok_xml + 0.5*ok_ast)
    return scores

def r_qiskit_import(prompts, completions, **_):
    scores = []
    for c in completions:
        ans = extract_answer(c[0]["content"])
        ok, why = is_qiskit_import_ok(ans)
        scores.append(1.0 if ok else (0.2 if why=="missing_qiskit_import" else 0.0))
    return scores

def r_compile_and_sim(prompts, completions, **_):
    """Reward if the function compiles, returns a circuit, and simulates."""
    scores = []
    for c in completions:
        ans = extract_answer(c[0]["content"])
        ok_ast, _ = safe_parse_func(ans)
        if not ok_ast:
            scores.append(0.0); continue
        fn, _ = build_callable(ans)
        if fn is None:
            scores.append(0.0); continue
        is_circ, qc = try_make_circuit(fn)
        if not is_circ:
            scores.append(0.0); continue
        ok_sv, sv = simulate_statevector(qc)
        ok_ct, ct = simulate_counts(qc)  
        scores.append( (1.0 if ok_sv else 0.0) + (0.5 if ok_ct else 0.0) )
    return scores

def r_fidelity_vs_reference(prompts, completions, references, **_):
    """If reference produces a circuit/statevector, compare state fidelity."""
    scores = []
    for c, ref_src in zip(completions, references):
        r_fn, _ = reference_adapter(ref_src)
        if r_fn is None:
            scores.append(0.0); continue
        r_ok_c, r_qc = try_make_circuit(r_fn)
        if not r_ok_c:
            scores.append(0.0); continue
        r_ok_sv, r_sv = simulate_statevector(r_qc)
        if not r_ok_sv:
            scores.append(0.0); continue

        ans = extract_answer(c[0]["content"])
        ok_ast, _ = safe_parse_func(ans)
        if not ok_ast:
            scores.append(0.0); continue
        fn, _ = build_callable(ans)
        if fn is None:
            scores.append(0.0); continue
        ok_c, qc = try_make_circuit(fn)
        if not ok_c:
            scores.append(0.0); continue
        ok_sv, sv = simulate_statevector(qc)
        if not ok_sv:
            scores.append(0.0); continue

        fid = float(state_fidelity(r_sv, sv))
        scores.append(2.0 * (fid ** 0.5))
    return scores

def r_resource_efficiency(prompts, completions, **_):
    """Prefer shallower, smaller circuits (depth & gate count)."""
    scores = []
    for c in completions:
        ans = extract_answer(c[0]["content"])
        ok_ast, _ = safe_parse_func(ans)
        if not ok_ast:
            scores.append(0.0); continue
        fn, _ = build_callable(ans)
        if fn is None:
            scores.append(0.0); continue
        ok_c, qc = try_make_circuit(fn)
        if not ok_c:
            scores.append(0.0); continue
        depth = qc.depth() or 1
        gates = sum(1 for _ in qc.data) or 1
        val = 1.0 / (1.0 + math.log1p(depth) + 0.5*math.log1p(gates))
        scores.append(val)
    return scores

training_args = GRPOConfig(
    use_vllm = False,
    learning_rate = LR,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 5,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_generations = NUM_GENERATIONS,
    max_prompt_length = MAX_PROMPT_LEN,
    max_completion_length = MAX_COMPLETION_LEN,
    num_train_epochs = EPOCHS,
    save_steps = 500,
    max_grad_norm = 0.2,
    report_to = ("none" if not USE_WANDB else "wandb"),
    output_dir = OUTPUT_DIR,
)


def _collate(ds):
    return {
        "prompts": ds["prompt"],
        "references": ds["reference"],
    }
train_inputs = _collate(dataset)

PatchDPOTrainer()  
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        r_format,
        r_qiskit_import,
        r_compile_and_sim,
        lambda prompts, completions, **kw: r_fidelity_vs_reference(prompts, completions, train_inputs["references"], **kw),
        r_resource_efficiency,
    ],
    args = training_args,
    train_dataset = dataset,  
)

trainer.train()

from unsloth import FastLanguageModel as FLM
FLM.for_inference(model)
lora_dir = os.path.join(OUTPUT_DIR, "lora_model")
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
print(f"[SAVED] LoRA adapters to {lora_dir}")

MERGE = False
if MERGE and MODEL_NAME:
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map="auto")
    lora = PeftModel.from_pretrained(base, lora_dir)
    merged = lora.merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"[MERGED] Full model saved to {MERGED_DIR}")
