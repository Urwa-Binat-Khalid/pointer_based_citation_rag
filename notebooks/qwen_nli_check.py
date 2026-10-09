"""
Qwen 2.5 7B: NLI check on the answers from qwen_run_systems_a_b.py.
Run qwen_setup_helpers.py first. It defines model and gen.
For every answer, the same Qwen model rewrites question and answer as one claim.
A separate DeBERTa NLI model then checks the claim against the pointed sentences
(System B) or the copied quote (System A).
Produces results_fresh300_C_qwen.json and prints the accept rule tables.
Resumes from the saved file if interrupted.
"""
import json, os, torch
from transformers import AutoTokenizer as AT, AutoModelForSequenceClassification as AMS

assert "model" in globals() and "gen" in globals(), "run qwen_setup_helpers.py first"

# Same text as CLAIM_SYSTEM in src/prompts.py
CLAIM_SYSTEM = (
    "Rewrite a yes/no question and its answer as ONE short declarative "
    "sentence stating the claim. If the answer is yes, state the claim as "
    "true. If the answer is no, state clearly that the claim does NOT hold "
    "(for example: there was no significant difference). "
    "Output only the sentence, nothing else."
)

NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
nli_tok = AT.from_pretrained(NLI)
nli_model = AMS.from_pretrained(NLI).to("cuda").eval()
labels = [nli_model.config.id2label[i].lower() for i in range(3)]

def nli_check(premise, hypothesis):
    enc = nli_tok(premise, hypothesis, truncation=True, max_length=512,
                  return_tensors="pt").to("cuda")
    with torch.no_grad():
        probs = torch.softmax(nli_model(**enc).logits, dim=-1)[0]
    return labels[int(probs.argmax())]

# In the original run the questions file was pubmedqa_fresh300.json
# (build_dataset.py writes the same questions to data/fresh300.json).
fresh = {it["id"]: it for it in json.load(open("pubmedqa_fresh300.json"))}
# Answers from qwen_run_systems_a_b.py. In the original run this file was read
# from a Kaggle input folder. Change the path if yours is elsewhere.
base = json.load(open("results_fresh300.json"))
OUT = "results_fresh300_C_qwen.json"
done = {r["id"]: r for r in json.load(open(OUT))} if os.path.exists(OUT) else {}
print("already done:", len(done))

def check(question, ans, premise):
    if ans in ("yes", "no") and premise.strip():
        claim, ct = gen(CLAIM_SYSTEM, f"Question: {question}\nAnswer: {ans}", 60)
        return nli_check(premise, claim.strip()), ct
    return "no_evidence", 0

for idx, r in enumerate(base):
    if r["id"] in done:
        continue
    item = fresh[r["id"]]
    premiseB = " ".join(item["doc"][p] for p in r["B_pointers"] if p in item["doc"])
    labB, ctB = check(item["question"], r["B_answer"], premiseB)
    labA, ctA = check(item["question"], r["A_answer"], r["A_evidence"])
    done[r["id"]] = {**r, "B_nli": labB, "B_claim_tokens": ctB,
                     "A_nli": labA, "A_claim_tokens": ctA}
    if (idx + 1) % 10 == 0:
        print("done", idx + 1)
        json.dump(list(done.values()), open(OUT, "w"), indent=2)

rows = [done[r["id"]] for r in base]
json.dump(rows, open(OUT, "w"), indent=2)
N = len(rows)

def summarize(tag, name):
    ans, nli = f"{tag}_answer", f"{tag}_nli"
    tok_total = sum(r[f"{tag}_tokens"] + r[f"{tag}_claim_tokens"] for r in rows) / N
    tok_ans = sum(r[f"{tag}_tokens"] for r in rows) / N
    corr = [r for r in rows if r[ans] == r["gold"]]
    wrg = [r for r in rows if r[ans] != r["gold"]]
    print(f"\n=== {name} ===")
    print(f"tokens per answer: {tok_ans:.1f} | with claim step: {tok_total:.1f}")
    for lab in ("entailment", "neutral", "contradiction"):
        c = sum(r[nli] == lab for r in corr)
        w = sum(r[nli] == lab for r in wrg)
        print(f"  {lab:14s} correct {c/len(corr):.0%} | wrong {w/max(len(wrg),1):.0%}")
    for rule, fn in (("accept everything", lambda r: True),
                     ("accept unless contradiction", lambda r: r[nli] != "contradiction"),
                     ("accept only entailment", lambda r: r[nli] == "entailment")):
        s = [r for r in rows if fn(r)]
        k = len(s)
        acc = sum(r[ans] == r["gold"] for r in s) / k if k else 0
        wa = sum(r[ans] != r["gold"] for r in s)
        wr = sum(1 for r in wrg if not fn(r))
        cr = sum(1 for r in corr if not fn(r))
        print(f"  {rule:28s} coverage {k}/{N} ({k/N:.0%}) | accuracy {acc:.1%} | wrong accepted {wa} | "
              f"wrong rejected {wr}/{len(wrg)} | correct rejected {cr}/{len(corr)}")

summarize("B", "C: pointer plus NLI (Qwen)")
summarize("A", "A plus NLI: copied quote plus the same checker (Qwen)")
print(f"\ncitation validity: A quotes found in document {sum(r['A_in_doc'] for r in rows)/N:.1%} | B invalid pointers {sum(len(r['B_invalid'])>0 for r in rows)}")
