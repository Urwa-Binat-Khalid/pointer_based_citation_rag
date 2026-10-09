#File 6 is notebooks/granite_run_systems_a_b.py
"""
Granite 3.3 8B Instruct: run System A (copied quote) and System B (pointer)
on the fresh 300 questions. Self contained: loads the model, defines the helpers,
runs both systems and prints accuracy, sign tests and bootstrap intervals.
Needs: pip install -U "bitsandbytes>=0.46.1" transformers accelerate
"""
import json, os, re, random, torch
from math import comb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "ibm-granite/granite-3.3-8b-instruct"
TAG = MODEL.split("/")[-1]

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")

# Same text as SYSTEM_B and SYSTEM_A in src/prompts.py
SYSTEM_B = (
    "You answer biomedical yes/no questions using ONLY the numbered "
    "sentences provided. Do not write any quotes or citations in text. "
    "Return only valid JSON in this exact format: "
    '{"answer": "yes" or "no", "pointers": ["S0", "S3"]} '
    "You must choose yes or no. Answer yes if the study results support the "
    "claim in the question. Answer no if the results contradict it or show "
    "no effect or no difference. "
    "The pointers must be the IDs of the sentences that most directly state "
    "the study result. Use at most 3 pointers and only IDs that exist."
)
SYSTEM_A2 = (
    "You answer biomedical yes/no questions using ONLY the numbered sentences provided. "
    "Return only valid JSON in this exact format: "
    '{"answer": "yes" or "no", "evidence": "the single most relevant sentence copied word for word from the document, without its ID"} '
    "Answer yes if the study results support the claim in the question. "
    "Answer no if the results contradict it or show no effect or no difference."
)

def gen(system, user, max_new_tokens=250):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    new = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True), len(new)

def numbered_doc(item):
    return "\n".join(f"{sid}: {t}" for sid, t in item["doc"].items())

def plain_doc(item):
    return " ".join(item["doc"].values())

def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0))
    except Exception:
        a = re.search(r'"answer"\s*:\s*"(\w+)"', text)
        e = re.search(r'"evidence"\s*:\s*"(.*)', text, re.DOTALL)
        return {"answer": a.group(1) if a else None,
                "evidence": e.group(1).strip('"} \n') if e else "",
                "pointers": []}

def squash(s):
    return re.sub(r"\s+", "", (s or "").lower())

# In the original run this file was called pubmedqa_fresh300.json.
# build_dataset.py writes the same questions to data/fresh300.json.
fresh = json.load(open("pubmedqa_fresh300.json"))
OUT = f"results_fresh300_{TAG}.json"
done = {r["id"]: r for r in json.load(open(OUT))} if os.path.exists(OUT) else {}
print("model:", MODEL, "| already done:", len(done))

for idx, item in enumerate(fresh):
    if item["id"] in done:
        continue
    q = f"Question: {item['question']}\n\nDocument:\n" + numbered_doc(item)
    rawB, tB = gen(SYSTEM_B, q, 150)
    rawA, tA = gen(SYSTEM_A2, q, 250)
    dB, dA = parse_json(rawB), parse_json(rawA)
    ptrs = dB.get("pointers", []) or []
    ev = re.sub(r"^\s*S\d+\s*:\s*", "", dA.get("evidence", "") or "")
    done[item["id"]] = {
        "id": item["id"], "gold": item["gold_answer"],
        "B_answer": dB.get("answer"), "B_pointers": ptrs,
        "B_invalid": [p for p in ptrs if p not in item["doc"]], "B_tokens": tB,
        "A_answer": dA.get("answer"), "A_evidence": ev,
        "A_in_doc": bool(squash(ev)) and squash(ev) in squash(plain_doc(item)),
        "A_tokens": tA,
    }
    if (idx + 1) % 10 == 0:
        print("done", idx + 1)
        json.dump(list(done.values()), open(OUT, "w"), indent=2)

rows = [done[it["id"]] for it in fresh]
json.dump(rows, open(OUT, "w"), indent=2)

N = len(rows)
gold = [r["gold"] for r in rows]
okA = [r["A_answer"] == r["gold"] for r in rows]
okB = [r["B_answer"] == r["gold"] for r in rows]

def balanced(ok, idx):
    vals = []
    for g in ("yes", "no"):
        s = [i for i in idx if gold[i] == g]
        if s:
            vals.append(sum(ok[i] for i in s) / len(s))
    return sum(vals) / len(vals)

def sign_test(b, a):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

idx_all = list(range(N))
print(f"\n{TAG} | N = {N} | always yes baseline: {sum(g == 'yes' for g in gold)/N:.0%}")
print(f"raw accuracy       A numbered quote {sum(okA)/N:.1%} | B pointers {sum(okB)/N:.1%}")
print(f"balanced accuracy  A {balanced(okA, idx_all):.1%} | B {balanced(okB, idx_all):.1%}")
print(f"share answering yes: A {sum(r['A_answer']=='yes' for r in rows)/N:.0%} | B {sum(r['B_answer']=='yes' for r in rows)/N:.0%}")
print(f"answers that were not yes or no: A {sum(r['A_answer'] not in ('yes','no') for r in rows)} | B {sum(r['B_answer'] not in ('yes','no') for r in rows)}")
for g in ("yes", "no"):
    s = [i for i in idx_all if gold[i] == g]
    b = sum(okB[i] and not okA[i] for i in s)
    a = sum(okA[i] and not okB[i] for i in s)
    print(f"gold {g} ({len(s)}): A {sum(okA[i] for i in s)} right | B {sum(okB[i] for i in s)} right | only B {b}, only A {a}, p = {sign_test(b, a):.4f}")
only_B = sum(b and not a for a, b in zip(okA, okB))
only_A = sum(a and not b for a, b in zip(okA, okB))
print(f"overall: only B right {only_B}, only A right {only_A}, p = {sign_test(only_B, only_A):.4f}")

random.seed(0)
d_raw, d_bal, d_tok = [], [], []
for _ in range(2000):
    s = [random.randrange(N) for _ in range(N)]
    d_raw.append(sum(okB[i] for i in s) / N - sum(okA[i] for i in s) / N)
    d_bal.append(balanced(okB, s) - balanced(okA, s))
    d_tok.append(sum(rows[i]["A_tokens"] - rows[i]["B_tokens"] for i in s) / N)

def ci(v):
    v = sorted(v)
    return round(v[50], 3), round(v[1949], 3)

print("\n95% bootstrap intervals")
print("raw accuracy, B minus A:       ", round(sum(okB)/N - sum(okA)/N, 3), ci(d_raw))
print("balanced accuracy, B minus A:  ", round(balanced(okB, idx_all) - balanced(okA, idx_all), 3), ci(d_bal))
print("answer tokens saved, A minus B:", round(sum(r['A_tokens'] - r['B_tokens'] for r in rows)/N, 1), ci(d_tok))
print(f"avg tokens: A {sum(r['A_tokens'] for r in rows)/N:.1f} | B {sum(r['B_tokens'] for r in rows)/N:.1f}")
print(f"A quotes found in document: {sum(r['A_in_doc'] for r in rows)/N:.1%} | B items with invalid pointers: {sum(len(r['B_invalid'])>0 for r in rows)}")
