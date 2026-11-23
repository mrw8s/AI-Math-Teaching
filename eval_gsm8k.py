import json, os, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm_utils import build_prompt_text, PREFILL, reference_value, score

# ------------------------------------------------------------------ config ----
MODEL_NAME   = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = None          # e.g. "/kaggle/working/rl_run/checkpoint-500" for SFT/RL models
TAG          = "base"        # names the output files: base / sft / rl
N_TEST       = 500           # size of the fixed test subset (first N test problems). Try 20 first!
N_PASSK      = 200           # problems used for the sampled / pass@k evaluation (subset of N_TEST)
K            = 8             # samples per problem for pass@k
TEMP         = 0.8
MAX_NEW      = 512
BATCH_GREEDY = 32            # prompts per batch for greedy decoding
BATCH_SAMPLE = 8             # prompts per batch for sampling (x K sequences each)
OUT_DIR      = "/kaggle/working"
SEED         = 0
# -------------------------------------------------------------------------------

torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.padding_side = "left"                      # required for batched generation
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda")
if ADAPTER_PATH:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

test = load_dataset("openai/gsm8k", "main")["test"].select(range(N_TEST))
questions = [ex["question"] for ex in test]
refs = [reference_value(ex["answer"]) for ex in test]
prompts = [build_prompt_text(tok, q) for q in questions]


@torch.no_grad()
def generate(batch_prompts, sample, n=1):
    enc = tok(batch_prompts, return_tensors="pt", padding=True).to("cuda")
    kw = dict(max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id,
              repetition_penalty=1.0, top_p=1.0, top_k=0)   # override Qwen's default sampling settings
    if sample:
        kw.update(do_sample=True, temperature=TEMP, num_return_sequences=n)
    else:
        kw.update(do_sample=False)
    out = model.generate(**enc, **kw)
    gen = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    lens = (gen != tok.pad_token_id).sum(dim=1).tolist()
    # the prompt ended with the prefill, so add it back before scoring
    return [PREFILL + t for t in texts], lens


records = []

# ------------------------------------------------ greedy (pass@1, one answer) ----
t0 = time.time()
greedy = []
for i in range(0, N_TEST, BATCH_GREEDY):
    texts, lens = generate(prompts[i:i + BATCH_GREEDY], sample=False)
    for j, (resp, L) in enumerate(zip(texts, lens)):
        idx = i + j
        sc = score(resp, refs[idx])
        greedy.append((sc, L))
        records.append(dict(mode="greedy", idx=idx, sample=0, response=resp, length=L,
                            reference=str(refs[idx]), **sc))
    print(f"greedy {min(i + BATCH_GREEDY, N_TEST)}/{N_TEST}  ({time.time() - t0:.0f}s)")

# ------------------------------------------ sampled (mean accuracy, pass@k) ----
sampled = []  # per problem: list of (score, length)
t1 = time.time()
for i in range(0, N_PASSK, BATCH_SAMPLE):
    batch = prompts[i:i + BATCH_SAMPLE]
    texts, lens = generate(batch, sample=True, n=K)
    for j in range(len(batch)):
        idx = i + j
        group = []
        for k in range(K):
            resp, L = texts[j * K + k], lens[j * K + k]
            sc = score(resp, refs[idx])
            group.append((sc, L))
            records.append(dict(mode="sample", idx=idx, sample=k, response=resp, length=L,
                                reference=str(refs[idx]), **sc))
        sampled.append(group)
    print(f"sampled {min(i + BATCH_SAMPLE, N_PASSK)}/{N_PASSK}  ({time.time() - t1:.0f}s)")


def mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


flat = [x for g in sampled for x in g]
summary = {
    "model": MODEL_NAME, "adapter": ADAPTER_PATH, "tag": TAG,
    "n_test": N_TEST, "n_passk": N_PASSK, "k": K, "temperature": TEMP, "max_new_tokens": MAX_NEW,
    "gpu": torch.cuda.get_device_name(0),
    "greedy": {
        "strict_acc":  mean([s["strict_correct"] for s, _ in greedy]),
        "lenient_acc": mean([s["lenient_correct"] for s, _ in greedy]),
        "format_rate": mean([s["format_ok"] for s, _ in greedy]),
        "avg_len_tokens": mean([L for _, L in greedy]),
    },
    "sampled": {
        "pass@1_strict":   mean([s["strict_correct"] for s, _ in flat]),
        "pass@1_lenient":  mean([s["lenient_correct"] for s, _ in flat]),
        f"pass@{K}_strict":  mean([any(s["strict_correct"] for s, _ in g) for g in sampled]),
        f"pass@{K}_lenient": mean([any(s["lenient_correct"] for s, _ in g) for g in sampled]),
        "format_rate": mean([s["format_ok"] for s, _ in flat]),
        "avg_len_tokens": mean([L for _, L in flat]),
        # share of problems where all K samples got the same lenient result (no GRPO signal)
        "all_same_groups": mean([len({s["lenient_correct"] for s, _ in g}) == 1 for g in sampled]),
    },
}

with open(f"{OUT_DIR}/{TAG}_responses.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
with open(f"{OUT_DIR}/{TAG}_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))