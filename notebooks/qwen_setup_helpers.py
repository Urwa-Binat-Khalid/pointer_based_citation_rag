#File 3
"""
Qwen setup and helpers. Run this before qwen_run_systems_a_b.py.
Loads Qwen 2.5 7B Instruct in 4 bit and defines the helper functions.
Needs: pip install -U "bitsandbytes>=0.46.1" transformers accelerate
"""
import json, os, re, torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

gc.collect(); torch.cuda.empty_cache()

MODEL = "Qwen/Qwen2.5-7B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")

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
SYSTEM_A = (
    "You answer biomedical yes/no questions using ONLY the sentences provided. "
    "Return only valid JSON in this exact format: "
    '{"answer": "yes" or "no", "evidence": "the single most relevant sentence copied word for word from the document"} '
    "Answer yes if the study results support the claim in the question. "
    "Answer no if the results contradict it or show no effect or no difference."
)

def gen(system, user, max_new_tokens=250):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
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
