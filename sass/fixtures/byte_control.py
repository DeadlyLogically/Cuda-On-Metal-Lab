from pathlib import Path
import re
import struct
import sys


def instruction_bytes(base, operation):
    pattern = (
        r"/\*00c0\*/\s+" + operation + r" R9, R4, R3 ;\s*"
        r"/\* 0x([a-f0-9]{16}) \*/\s*/\* 0x([a-f0-9]{16}) \*/"
    )
    word = re.search(pattern, base.with_suffix(".hex.sass").read_text())
    if not word:
        raise ValueError(f"Expected '{operation} R9, R4, R3' at PC 0xc0")
    pair = struct.pack("<QQ", *(int(value, 16) for value in word.groups()))
    data = base.with_suffix(".cubin").read_bytes()
    if data.count(pair) != 1:
        raise ValueError(f"Instruction encoding is not unique in {base.name}")
    return pair


build = Path(sys.argv[1])
addition = build / "vectorAdd.sm75.cubin"
multiply = build / "elementMultiply.sm75.cubin"
old = instruction_bytes(addition, "FADD")
new = instruction_bytes(multiply, "FMUL")
(build / "sameNameMultiply.sm75.cubin").write_bytes(
    addition.read_bytes().replace(old, new, 1)
)
