#File 4 
"""
Qwen 2.5 7B: run System A (copied quote) and System B (pointer) on the fresh 300 questions.
Run qwen_setup_helpers.py first. It defines model, gen, numbered_doc, plain_doc,
parse_json, squash and SYSTEM_B.
Produces results_fresh300.json and prints accuracy, sign tests and bootstrap intervals.
Resumes from the saved file if interrupted.
"""
import json, os, re, random
from math import comb

# Same text as SYSTEM_A in src/prompts.py (numbered document, copy one sentence)
SYSTEM_A2 = (
    "You answer biomedical yes/no questions using ONLY the numbered sentences provided. "
    "Return only valid JSON in this exact format: "
    '{"answer": "yes" or "no", "evidence": "the single most relevant sentence copied word for word from the document, without its ID"} '
    "Answer yes if the study results support the claim in the question. "
    "Answer no if the results contradict it or show no effect or no difference."
)

assert "model" in globals() and "SYSTEM_B" in globals() and "gen" in globals() \
    and "parse_json" in globals(), "run qwen_setup_helpers.py first"

FRESH_FILE = "pubmedqa_fresh300.json"
fresh = json.load(open(FRESH_FILE))
OUT = "results_fresh300.json"
done = {r["id"]: r for r in json.load(open(OUT))} if os.path.exists(OUT) else {}
print("already done:", len(done))

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

# ---------- summary ----------
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
print(f"\nFresh test, N = {N} | always yes baseline: {sum(g == 'yes' for g in gold)/N:.0%}")
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
print("raw accuracy, B minus A:      ", round(sum(okB)/N - sum(okA)/N, 3), ci(d_raw))
print("balanced accuracy, B minus A: ", round(balanced(okB, idx_all) - balanced(okA, idx_all), 3), ci(d_bal))
print("answer tokens saved, A minus B:", round(sum(r['A_tokens'] - r['B_tokens'] for r in rows)/N, 1), ci(d_tok))
print(f"avg tokens: A {sum(r['A_tokens'] for r in rows)/N:.1f} | B {sum(r['B_tokens'] for r in rows)/N:.1f}")
print(f"A quotes found in document: {sum(r['A_in_doc'] for r in rows)/N:.1%} | B items with invalid pointers: {sum(len(r['B_invalid'])>0 for r in rows)}")
