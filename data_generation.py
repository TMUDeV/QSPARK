import os
import ast
import re
import json
import time
import hashlib
import requests
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


GITHUB_API = "https://api.github.com/search/code"
RAW_BASE = "https://raw.githubusercontent.com"
OUTDIR = "qiskit_repos"
DATASET_PATH = "qiskit_humaneval_dataset.json"
SIMILARITY_THRESHOLD = 0.95

def search_qiskit_files(token, max_files=100, per_page=100):
    headers = {"Authorization": f"token {token}"}
    query = '("import qiskit" OR "from qiskit") language:python'
    params = {"q": query, "per_page": per_page, "page": 1}
    results = []

    while len(results) < max_files:
        print(f"[INFO] Searching page {params['page']} ...")
        r = requests.get(GITHUB_API, headers=headers, params=params)
        if r.status_code != 200:
            print(f"[ERROR] GitHub API {r.status_code}: {r.text}")
            break
        items = r.json().get("items", [])
        if not items:
            break
        results.extend(items)
        if len(items) < per_page:
            break
        params["page"] += 1
        time.sleep(1.5)
    print(f"[INFO] Found {len(results)} candidate files.")
    return results[:max_files]

def download_file(item):
    repo = item["repository"]["full_name"]
    path = item["path"]
    branch = item["repository"]["default_branch"]
    raw_url = f"{RAW_BASE}/{repo}/{branch}/{path}"

    os.makedirs(OUTDIR, exist_ok=True)
    filename = os.path.join(OUTDIR, repo.replace("/", "_") + "__" + path.replace("/", "_"))
    if not filename.endswith(".py"):
        filename += ".py"

    try:
        r = requests.get(raw_url, timeout=10)
        if r.status_code == 200 and "qiskit" in r.text.lower():
            with open(filename, "w", encoding="utf-8") as f:
                f.write(r.text)
            return filename
    except Exception as e:
        print(f"[WARN] Failed {repo}/{path}: {e}")
    return None

def extract_functions_from_code(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    functions = []
    lines = code.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            start = node.lineno - 1
            end = max(getattr(node, "end_lineno", start + 1), start + 1)
            snippet = "\n".join(lines[start:end])
            docstring = ast.get_docstring(node) or ""
            if len(snippet.strip()) > 25:
                functions.append({
                    "name": node.name,
                    "code": snippet,
                    "docstring": docstring
                })
    return functions


def is_valid_function(func):
    if func["name"].startswith("test_"):
        return False
    try:
        ast.parse(func["code"])
        return True
    except Exception:
        return False

def deduplicate(entries):
    unique = []
    seen_hashes = {}
    for row in entries:
        code = row["canonical_solution"].strip()
        h = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue

        similar = False
        for e in unique:
            overlap = len(set(e["canonical_solution"].split()) & set(code.split())) / max(1, len(set(code.split())))
            if overlap >= SIMILARITY_THRESHOLD:
                similar = True
                break
        if not similar:
            seen_hashes[h] = True
            unique.append(row)
    return unique

def analyze_difficulty(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "unknown"

    num_stmts = sum(isinstance(n, (ast.Assign, ast.If, ast.For, ast.While, ast.FunctionDef, ast.Return)) for n in ast.walk(tree))
    num_calls = sum(isinstance(n, ast.Call) for n in ast.walk(tree))
    num_branches = sum(isinstance(n, (ast.If, ast.For, ast.While, ast.Try)) for n in ast.walk(tree))

    if num_stmts <= 5 and num_calls <= 2 and num_branches == 0:
        return "basic"
    elif 5 < num_stmts <= 15 or num_calls > 2 or num_branches > 0:
        return "intermediate"
    else:
        return "advanced"

def build_qiskit_dataset(token, max_files=100):
    items = search_qiskit_files(token, max_files)
    print(f"[INFO] Downloading files ...")

    downloaded = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(download_file, i) for i in items]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            r = fut.result()
            if r:
                downloaded.append(r)

    print(f"[INFO] Extracting functions from {len(downloaded)} files ...")
    dataset = []
    task_id = 0
    for fpath in tqdm(downloaded):
        try:
            code = Path(fpath).read_text(encoding="utf-8")
            funcs = extract_functions_from_code(code)
            for func in funcs:
                if not is_valid_function(func):
                    continue
                diff = analyze_difficulty(func["code"])
                dataset.append({
                    "task_id": f"qiskitHumanEval/{task_id}",
                    "prompt": f"# {func['docstring']}\n\n# Implement function `{func['name']}` below.",
                    "canonical_solution": func["code"],
                    "test": f"def check(candidate):\n    # TODO: write tests for `{func['name']}`\n    pass\n",
                    "entry_point": func["name"],
                    "difficulty_scale": diff,
                })
                task_id += 1
        except Exception as e:
            print(f"[WARN] Skipping {fpath}: {e}")

    print(f"[INFO] Deduplicating {len(dataset)} entries ...")
    dataset = deduplicate(dataset)
    print(f"[INFO] {len(dataset)} unique tasks remain.")

    difficulty_counts = Counter(d["difficulty_scale"] for d in dataset)
    print(f"[STATS] Difficulty distribution: {dict(difficulty_counts)}")

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    print(f"[DONE] Dataset saved to {DATASET_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="GitHub personal access token")
    parser.add_argument("--max-files", type=int, default=100)
    args = parser.parse_args()

    build_qiskit_dataset(args.token, args.max_files)