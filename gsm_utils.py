import re
from fractions import Fraction

# ---------------------------------------------------------------- prompt ----
SYSTEM_PROMPT = (
    "Think step by step inside <think></think> tags, then give only the final "
    "number inside <answer></answer> tags."
)
FEWSHOT_Q = (
    "Tom has 5 apples. He buys 3 bags with 4 apples each, then gives away 6 "
    "apples. How many apples does he have now?"
)
FEWSHOT_A = (
    "<think>3 bags of 4 apples is 3*4=12 apples. He has 5+12=17 apples. "
    "After giving away 6, he has 17-6=11.</think><answer>11</answer>"
)
PREFILL = "<think>\n"  # added to the end of every prompt; add it back to the completion before scoring


def build_messages(question):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEWSHOT_Q},
        {"role": "assistant", "content": FEWSHOT_A},
        {"role": "user", "content": question},
    ]


def build_prompt_text(tokenizer, question):
    text = tokenizer.apply_chat_template(
        build_messages(question), tokenize=False, add_generation_prompt=True
    )
    return text + PREFILL


# ------------------------------------------------------------ extraction ----
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?|-?\.\d+")
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S)
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_FORMAT = re.compile(r"\s*<think>(.+?)</think>\s*<answer>([^<]+)</answer>\s*", re.S)
MIN_THINK_CHARS = 20  # reject empty / trivial reasoning


def to_number(s):
    """'$1,800' -> 1800, '1/2' and '0.5' -> same Fraction, '' -> None."""
    if s is None:
        return None
    s = s.strip()
    s = re.sub(r"\\d?frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", s)
    s = s.replace("$", "").replace(",", "").replace("%", "").replace(" ", "").rstrip(".")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def last_number(text):
    found = _NUMBER.findall(text or "")
    for tok in reversed(found):
        v = to_number(tok)
        if v is not None:
            return v
    return None


def reference_value(reference_solution):
    """GSM8K reference ends with '#### 18'."""
    return to_number(reference_solution.split("####")[-1])


def extract_strict(response):
    """Exactly one <answer> tag whose content is purely a number, else None."""
    tags = _ANSWER_TAG.findall(response)
    if len(tags) != 1:
        return None
    return to_number(tags[0])


def extract_lenient(response):
    """<answer> tag first, then \\boxed{}, then the last number in the text."""
    tags = _ANSWER_TAG.findall(response)
    if tags:
        v = to_number(tags[-1])
        if v is None:
            v = last_number(tags[-1])
        if v is not None:
            return v
    boxed = _BOXED.findall(response)
    if boxed:
        v = to_number(boxed[-1])
        if v is None:
            v = last_number(boxed[-1])
        if v is not None:
            return v
    return last_number(response)


def format_ok(response):
    """True only for <think>real reasoning</think><answer>number</answer>."""
    # each tag exactly once: stops "repeat the whole block" tricks
    for tag in ("<think>", "</think>", "<answer>", "</answer>"):
        if response.count(tag) != 1:
            return False
    m = _FORMAT.fullmatch(response)
    return bool(m) and len(m.group(1).strip()) >= MIN_THINK_CHARS


def score(response, ref_value):
    """The three numbers every evaluation and reward needs."""
    s = extract_strict(response)
    l = extract_lenient(response)
    return {
        "strict_correct": s is not None and s == ref_value,
        "lenient_correct": l is not None and l == ref_value,
        "format_ok": format_ok(response),
    }