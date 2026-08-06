"""Grade-1 English braille six-dot encoding utilities (63-class schema, decision D2).

Dot numbering within a cell:
    1 4
    2 5
    3 6

A cell's class id is the 6-bit mask where dot d contributes bit (d-1), so ids run
1..63 (0 = empty cell, which is not a class). This matches the Angelina 1..63 label
convention and the Unicode Braille Patterns block (char = U+2800 + mask).

Run `python scripts/common/braille.py` to print the full a-z table as a sanity check.
"""
from __future__ import annotations

# Grade-1 English letter -> dots present in the cell.
LETTER_TO_DOTS = {
    "a": (1,),            "b": (1, 2),          "c": (1, 4),
    "d": (1, 4, 5),       "e": (1, 5),          "f": (1, 2, 4),
    "g": (1, 2, 4, 5),    "h": (1, 2, 5),       "i": (2, 4),
    "j": (2, 4, 5),       "k": (1, 3),          "l": (1, 2, 3),
    "m": (1, 3, 4),       "n": (1, 3, 4, 5),    "o": (1, 3, 5),
    "p": (1, 2, 3, 4),    "q": (1, 2, 3, 4, 5), "r": (1, 2, 3, 5),
    "s": (2, 3, 4),       "t": (2, 3, 4, 5),    "u": (1, 3, 6),
    "v": (1, 2, 3, 6),    "w": (2, 4, 5, 6),    "x": (1, 3, 4, 6),
    "y": (1, 3, 4, 5, 6), "z": (1, 3, 5, 6),
}


def dots_to_class(dots) -> int:
    """(1,4,5) -> 25. Raises on empty or out-of-range dots."""
    mask = 0
    for d in dots:
        if not 1 <= int(d) <= 6:
            raise ValueError(f"dot out of range 1..6: {d}")
        mask |= 1 << (int(d) - 1)
    if not 1 <= mask <= 63:
        raise ValueError(f"empty/invalid cell from dots {dots!r}")
    return mask


def class_to_dots(class_id: int) -> tuple:
    if not 1 <= class_id <= 63:
        raise ValueError(f"class id out of range 1..63: {class_id}")
    return tuple(d for d in range(1, 7) if class_id & (1 << (d - 1)))


def dots_string_to_class(s: str) -> int:
    """'145' -> 25. A dot-string is digits 1..6, each dot at most once."""
    s = str(s).strip()
    if not s or any(c not in "123456" for c in s):
        raise ValueError(f"not a dot-string: {s!r}")
    return dots_to_class(tuple(int(c) for c in s))


def class_to_dots_string(class_id: int) -> str:
    return "".join(str(d) for d in class_to_dots(class_id))


def class_to_unicode(class_id: int) -> str:
    return chr(0x2800 + class_id)


LETTER_TO_CLASS = {ltr: dots_to_class(d) for ltr, d in LETTER_TO_DOTS.items()}
CLASS_TO_LETTER = {c: ltr for ltr, c in LETTER_TO_CLASS.items()}


def coco_categories() -> list:
    """The 63 COCO categories. id = dot mask, name = dot-string (e.g. '145'),
    with the Grade-1 letter and Unicode glyph attached for readability."""
    cats = []
    for cid in range(1, 64):
        cats.append({
            "id": cid,
            "name": class_to_dots_string(cid),
            "supercategory": "braille",
            "letter": CLASS_TO_LETTER.get(cid, ""),
            "unicode": class_to_unicode(cid),
        })
    return cats


if __name__ == "__main__":
    import sys
    try:  # Windows consoles default to cp1252 and choke on braille glyphs
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    print("letter  dots    class  glyph")
    for ltr in "abcdefghijklmnopqrstuvwxyz":
        c = LETTER_TO_CLASS[ltr]
        print(f"  {ltr}     {class_to_dots_string(c):<6} {c:<5} {class_to_unicode(c)}")
    print(f"\n{len(coco_categories())} categories (ids 1..63).")
