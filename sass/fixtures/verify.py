import hashlib
import json
import re
from pathlib import Path


root = Path(__file__).resolve().parent
build = root / "build"
original_source_hash = (
    "fe190d34eb7f7675d64c3a8589af4a44bc4398aa5a7d616b8d01370f28f8a745"
)
original_cubin_hash = "6e5eb269de727a5806cc4f77c2ba394e0b4e6af53f6f6400a7612121f18920e7"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instruction_counts(disassembly):
    tokens = re.findall(
        r"/\*[0-9a-f]+\*/\s+(?:@[!\w]+\s+)?([A-Z][A-Z0-9_.]+)\b", disassembly
    )
    return {op: tokens.count(op) for op in sorted(set(tokens))}


def inspect(kernel, source):
    path = build / (kernel + ".sm75.cubin")
    disassembly = path.with_suffix(".sass").read_text()
    ptx_dump = path.with_suffix(".embedded-ptx.txt")
    if "EF_CUDA_SM75" not in disassembly or ptx_dump.stat().st_size:
        raise ValueError(
            f"{kernel}: expected sm_75 with an empty cuobjdump --dump-ptx result"
        )
    original = root / "src" / (source + ".cu")
    expected_source = (
        original_source_hash
        if source == "vectorAdd_kernel" and not original.exists()
        else digest(original)
    )
    return {
        "kernel": kernel,
        "cubin_sha256": digest(path),
        "cubin_bytes": path.stat().st_size,
        "source_sha256": expected_source,
        "embedded_ptx_dump_bytes": ptx_dump.stat().st_size,
        "instruction_counts": instruction_counts(disassembly),
    }


records = {
    name: inspect(name, src)
    for name, src in [
        ("vectorAdd", "vectorAdd_kernel"),
        ("tripleSum", "tripleSum"),
        ("elementMultiply", "elementMultiply"),
        ("atomicSum", "atomicSum"),
    ]
}
if records["vectorAdd"]["source_sha256"] != original_source_hash:
    raise ValueError("upstream CUDA source identity changed")
if records["vectorAdd"]["cubin_sha256"] != original_cubin_hash:
    raise ValueError("vectorAdd differs from the original test artifact")
for name, required in [("tripleSum", "FADD"), ("elementMultiply", "FMUL")]:
    if required not in records[name]["instruction_counts"]:
        raise ValueError(f"{name}: compiler did not emit {required}")
if records["tripleSum"]["instruction_counts"]["FADD"] != 2:
    raise ValueError("tripleSum must compose two FADD instructions")
extra = set(records["tripleSum"]["instruction_counts"]).difference(
    records["vectorAdd"]["instruction_counts"]
)
if extra.difference({"NOP"}):
    raise ValueError("tripleSum added non-padding opcodes that vectorAdd does not use")
if not any(
    op.startswith(("RED", "ATOM")) for op in records["atomicSum"]["instruction_counts"]
):
    raise ValueError("negative fixture does not contain an atomic operation")
control = build / "sameNameMultiply.sm75.cubin"
original = build / "vectorAdd.sm75.cubin"
control_sass = control.with_suffix(".sass").read_text()
if control_sass != original.with_suffix(".sass").read_text().replace(
    "FADD R9, R4, R3", "FMUL R9, R4, R3"
):
    raise ValueError(
        "same-symbol control's disassembly changed beyond the selected instruction"
    )
if (
    control.with_suffix(".embedded-ptx.txt").stat().st_size
    or control.stat().st_size != original.stat().st_size
):
    raise ValueError("invalid byte-control experiment")
changed_bytes = [
    i
    for i, pair in enumerate(zip(original.read_bytes(), control.read_bytes()))
    if pair[0] != pair[1]
]
if not changed_bytes or changed_bytes[-1] - changed_bytes[0] >= 16:
    raise ValueError("modified bytes are outside the selected instruction")
records["sameNameMultiply"] = {
    "kernel": "vectorAdd",
    "oracle": "multiply",
    "origin": "one instruction replaced in vectorAdd",
    "cubin_sha256": digest(control),
    "cubin_bytes": control.stat().st_size,
    "embedded_ptx_dump_bytes": 0,
    "changed_file_byte_offsets": changed_bytes,
    "instruction_counts": instruction_counts(control_sass),
}
text = json.dumps(records, indent=2) + "\n"
(build / "manifest.json").write_text(text)
print(text)
