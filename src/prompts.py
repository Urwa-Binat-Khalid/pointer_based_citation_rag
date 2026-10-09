#File 2: src/prompts.py
"""
Frozen prompts for the pointer citation experiments.
Do not edit these if you want to reproduce the fresh test results.

SYSTEM_A      System A: the model copies the supporting sentence as text
SYSTEM_B      System B: the model outputs only sentence IDs (pointers)
CLAIM_SYSTEM  rewrites a question and answer as one claim for the NLI check
"""

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
    "You answer biomedical yes/no questions using ONLY the numbered sentences provided. "
    "Return only valid JSON in this exact format: "
    '{"answer": "yes" or "no", "evidence": "the single most relevant sentence copied word for word from the document, without its ID"} '
    "Answer yes if the study results support the claim in the question. "
    "Answer no if the results contradict it or show no effect or no difference."
)

CLAIM_SYSTEM = (
    "Rewrite a yes/no question and its answer as ONE short declarative "
    "sentence stating the claim. If the answer is yes, state the claim as "
    "true. If the answer is no, state clearly that the claim does NOT hold "
    "(for example: there was no significant difference). "
    "Output only the sentence, nothing else."
)


def numbered_doc(item):
    """The document as numbered sentences, one per line. Same input for A and B."""
    return "\n".join(f"{sid}: {text}" for sid, text in item["doc"].items())


def build_user_prompt(item):
    return f"Question: {item['question']}\n\nDocument:\n" + numbered_doc(item)
