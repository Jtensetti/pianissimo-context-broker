"""Conservative text-derived pronunciation proximity, not acoustic evidence.

This small Swedish-oriented approximation keeps vowels and consonant order.
It deliberately rejects unsupported/uncertain forms rather than using a coarse
Soundex key or accepting the model's own claim about pronunciation.
"""
from __future__ import annotations

from functools import lru_cache
from difflib import SequenceMatcher
import re
import unicodedata


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


@lru_cache(maxsize=2048)
def pronunciation_key(token: str) -> str:
    # A small explicit initialism reading. Unknown acronyms are not expanded
    # speculatively. More readings require documented regression examples.
    if token == "ai":
        return "aj"
    if not re.fullmatch(r"[a-zåäöé]+", token):
        return ""
    token = token.replace("é", "e").replace("w", "v").replace("z", "s")
    token = re.sub(r"skj|stj|sch|sh|sj", "ʃ", token)
    token = re.sub(r"tj|kj", "ɕ", token)
    token = re.sub(r"ck", "k", token)
    token = re.sub(r"c(?=[eiyäö])", "s", token).replace("c", "k")
    token = token.replace("q", "k").replace("x", "ks")
    # Spelling-based ASR slips commonly duplicate/drop consonant letters.
    # Keep vowels: collapsing them into one class admits unrelated words.
    return re.sub(r"([^aeiouyåäö])\1+", r"\1", token)


def _distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def phonetically_close(source: str, replacement: str) -> bool:
    left, right = _tokens(source), _tokens(replacement)
    # Compare only the changed span. Unchanged surrounding words must not
    # inflate similarity for 'jag är ledsen' -> 'jag är deprimerad'.
    while left and right and left[0] == right[0]:
        left.pop(0)
        right.pop(0)
    while left and right and left[-1] == right[-1]:
        left.pop()
        right.pop()
    if not left and not right:
        return True
    if not left or not right:
        return False  # Never insert/delete a whole spoken word.
    left_keys = [pronunciation_key(t) for t in left]
    right_keys = [pronunciation_key(t) for t in right]
    if not all(left_keys + right_keys):
        return False
    a, b = "".join(left_keys), "".join(right_keys)
    if a == b:
        return True
    if min(len(a), len(b)) < 5:
        return False  # One sound is too much uncertainty in a very short word.
    distance = _distance(a, b)
    return distance <= 2 and distance / max(len(a), len(b)) <= .20


def changes_stay_close(raw: str, candidate: str) -> bool:
    """Check each changed span against immutable ASR, preventing repair drift."""
    left, right = _tokens(raw), _tokens(candidate)
    matcher = SequenceMatcher(None, left, right, autojunk=False)
    return all(tag == "equal" or phonetically_close(" ".join(left[a:b]), " ".join(right[c:d]))
               for tag, a, b, c, d in matcher.get_opcodes())
