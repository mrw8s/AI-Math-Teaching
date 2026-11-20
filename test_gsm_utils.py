from fractions import Fraction
from gsm_utils import *

def A(x):  # wrap a final answer in a valid-format response
    return f"<think>some real reasoning goes here for the test</think><answer>{x}</answer>"

# number normalisation
assert to_number("18") == to_number("18.0") == to_number("18.00") == 18
assert to_number("$1,800") == 1800
assert to_number("1/2") == to_number("0.5") == Fraction(1, 2)
assert to_number("12%") == 12
assert to_number("18.") == 18
assert to_number("") is None and to_number("abc") is None and to_number("1/0") is None

# reference
assert reference_value("Janet sells ...\n#### 18") == 18
assert reference_value("blah\n#### 1,200") == 1200

# strict extractor
assert extract_strict(A("18")) == 18
assert extract_strict(A("$12")) == 12
assert extract_strict(A("1/2")) == Fraction(1, 2)
assert extract_strict(A("")) is None
assert extract_strict(A("18 dollars")) is None              # not purely a number
assert extract_strict(A("3") + A("18")) is None             # two answer tags -> hack, no credit
assert extract_strict("The answer is 18") is None
assert extract_strict("<answer>18</thought>") is None       # unclosed answer tag

# lenient extractor
assert extract_lenient("<think>x</thought><answer>2</answer>") == 2
assert extract_lenient("so \\(\\boxed{18}\\) dollars") == 18
assert extract_lenient("\\boxed{\\frac{1}{2}}") == Fraction(1, 2)
assert extract_lenient("\\boxed{18 \\text{ dollars}}") == 18
assert extract_lenient("It costs 5 then 7 so total 12.") == 12
assert extract_lenient("<answer>18 dollars</answer>") == 18
assert extract_lenient("no numbers here") is None

# format check
assert format_ok(A("18"))
assert not format_ok("<think>some real reasoning goes here for the test</thought><answer>18</answer>")
assert not format_ok("<think>hi</think><answer>18</answer>")            # trivial reasoning
assert not format_ok(A("3") + A("18"))                                   # repeated
assert not format_ok("<answer>18</answer>")

# combined score
ref = Fraction(18)
r = score(A("18"), ref);                     assert r == dict(strict_correct=True, lenient_correct=True, format_ok=True)
r = score("so \\boxed{18}", ref);            assert r == dict(strict_correct=False, lenient_correct=True, format_ok=False)
r = score(A("17"), ref);                     assert r == dict(strict_correct=False, lenient_correct=False, format_ok=True)
r = score(A("3") + A("18"), ref);            assert not r["strict_correct"] and not r["format_ok"]

print("all checker tests passed")