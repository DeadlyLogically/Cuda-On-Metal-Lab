#!/usr/bin/env python3

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


class Unsupported(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Unsupported(reason)


@dataclass(frozen=True)
class Kernel:
    name: str
    code_offset: int
    code_size: int
    base: int
    params: tuple


def region(data, offset, size):
    require(
        0 <= offset <= len(data) and 0 <= size <= len(data) - offset,
        "ELF record extends beyond input",
    )
    return data[offset : offset + size]


def parameter_info(data):
    fields, table, size = {}, None, None
    pos = 0
    while pos < len(data):
        fmt, attr, value = struct.unpack("<BBH", region(data, pos, 4))
        pos += 4
        if fmt == 4:
            payload = region(data, pos, value)
            pos += value
        else:
            require(fmt in {1, 2, 3}, f"unknown nv.info record format {fmt}")
            payload = b""
        if (fmt, attr) == (4, 0x0A):
            require(len(payload) == 8 and table is None, "invalid parameter bank")
            _, base, total = struct.unpack("<IHH", payload)
            table = (base, total)
        elif (fmt, attr) == (3, 0x19):
            require(size is None, "duplicate parameter size")
            size = value
        elif (fmt, attr) == (4, 0x17):
            require(len(payload) == 12, "invalid kernel parameter")
            _, ordinal, offset, flags = struct.unpack("<IHHI", payload)
            width = (flags >> 18) & 0x3FFF
            require(
                ordinal not in fields and width in (4, 8),
                "unsupported parameter width/ordinal",
            )
            fields[ordinal] = (offset, width)
    require(
        table is not None and table[1] == size and bool(fields),
        "inconsistent parameter bank",
    )
    require(set(fields) == set(range(len(fields))), "non-contiguous parameter ordinals")
    params = tuple(fields[i] for i in range(len(fields)))
    occupied = set()
    for offset, width in params:
        bytes_ = set(range(offset, offset + width))
        require(
            offset % width == 0 and offset + width <= size and not (occupied & bytes_),
            "overlapping, misaligned, or out-of-range kernel argument",
        )
        occupied.update(bytes_)
    return table[0], params


def inventory(data):
    h = struct.unpack("<16sHHIQQQIHHHHHH", region(data, 0, 64))
    require(
        h[0][:7] == b"\x7fELF\x02\x01\x01" and h[1:3] == (2, 190),
        "expected standalone little-endian ELF64 NVIDIA executable",
    )
    require(h[7] & 255 == 75, "only sm_75 Turing instructions are supported")
    require(h[11] == 64 and 0 < h[13] < h[12], "invalid section headers")
    headers = [
        struct.unpack("<IIQQQQIIQQ", region(data, h[6] + i * 64, 64))
        for i in range(h[12])
    ]
    strings = region(data, *headers[h[13]][4:6])
    sections = {}
    for hdr in headers:
        end = strings.find(b"\0", hdr[0])
        require(end >= hdr[0], "invalid ELF section name")
        name = strings[hdr[0] : end].decode("ascii")
        require(name not in sections, f"duplicate ELF section {name}")
        sections[name] = hdr
    require(
        not any(
            n.startswith((".nv_fatbin", ".nvFatBin", ".nv_debug_ptx")) for n in sections
        ),
        "expected standalone machine code without embedded PTX",
    )
    kernels = {}
    for section, hdr in sections.items():
        if not section.startswith(".text.") or hdr[5] == 0:
            continue
        name = section[6:]
        require(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name), f"unsupported symbol {name}"
        )
        metadata = sections.get(".nv.info." + name)
        require(
            metadata and metadata[7] == headers.index(hdr),
            f"missing metadata for {name}",
        )
        require(hdr[5] % 16 == 0 and hdr[2] & 4, f"invalid code section for {name}")
        region(data, *hdr[4:6])
        require(
            not any(
                s in sections for s in (".rel." + section[1:], ".rela." + section[1:])
            ),
            f"relocations in {section} need a linker",
        )
        base, params = parameter_info(region(data, *metadata[4:6]))
        kernels[name] = Kernel(name, hdr[4], hdr[5], base, params)
    require(kernels, "no supported CUDA entry points")
    return kernels


def instructions(listing, kernels, data):
    found = {}
    current = None
    entries = set()
    lines = iter(listing.splitlines())
    for line in lines:
        section = re.match(r"\s*\.section\s+(\.[^,\s]+)", line)
        if section:
            name = section[1]
            current = name[6:] if name.startswith(".text.") else None
            if current is not None:
                require(
                    current in kernels and current not in found,
                    f"unexpected text section {current}",
                )
                found[current] = ([], {})
            continue
        if current is None:
            continue
        code, labels = found[current]
        if re.search(r"\.other\s+" + re.escape(current) + r',@"STO_CUDA_ENTRY\b', line):
            entries.add(current)
        label = re.fullmatch(r"\s*([.\w$]+):\s*", line)
        if label:
            labels[label[1]] = len(code) * 16
            continue
        m = re.fullmatch(
            r"\s*/\*([\da-fA-F]+)\*/\s*(.*?)\s*;\s*/\* 0x([0-9a-fA-F]{16}) \*/\s*", line
        )
        if m:
            pc = int(m[1], 16)
            require(pc == len(code) * 16, "noncontiguous SASS instruction offsets")
            high = re.fullmatch(r"\s*/\* 0x([0-9a-fA-F]{16}) \*/\s*", next(lines, ""))
            require(high, f"missing high instruction word at {current}+0x{pc:x}")
            raw = struct.pack("<QQ", int(m[3], 16), int(high[1], 16))
            require(
                raw == region(data, kernels[current].code_offset + pc, 16),
                f"instruction word mismatch at {current}+0x{pc:x}",
            )
            code.append(m[2])
    require(found.keys() == kernels.keys(), "missing disassembled kernels")
    require(entries == kernels.keys(), "only CUDA kernel entry points are supported")
    for name, (code, _) in found.items():
        require(
            len(code) * 16 == kernels[name].code_size,
            f"incomplete disassembly of {name}",
        )
    return found


@dataclass(frozen=True)
class Value:
    kind: str
    name: str


class Lift:
    def __init__(self, kernel):
        self.k = kernel
        self.body = []
        self.decl = []
        self.regs = {}
        self.pairs = {}
        self.args = {}
        self.uses = Counter()
        self.exited = False

    def reg(self, kind):
        name = "%v" + str(len(self.decl))
        physical = {"addr": "u64"}.get(kind, kind)
        self.decl.append(f"    .reg .{physical} {name};")
        return Value(kind, name)

    def emit(self, op, *args):
        self.body.append("    " + op + (" " + ", ".join(args) if args else "") + ";")

    def make(self, kind, op, *args):
        out = self.reg(kind)
        self.emit(op, out.name, *args)
        return out

    def constant(self, operand):
        c = re.fullmatch(r"c\[(0x[0-9a-fA-F]+)\]\[(0x[0-9a-fA-F]+)\]", operand)
        require(c and int(c[1], 16) == 0, f"unsupported constant reference {operand}")
        return int(c[2], 16)

    def argument(self, offset, width):
        for i, (relative, size) in enumerate(self.k.params):
            if self.k.base + relative == offset and size == width:
                if i not in self.args:
                    self.args[i] = self.make(
                        f"u{width * 8}", f"ld.param.u{width * 8}", f"[arg{i}]"
                    )
                return self.args[i]
        raise Unsupported(
            f"unrecognized {width}-byte kernel argument at c[0][0x{offset:x}]"
        )

    def read(self, operand):
        if operand == "RZ":
            return Value("u32", "0")
        if re.fullmatch(r"0x[\da-fA-F]+|\d+", operand):
            value = int(operand, 0)
            require(value < 2**32, "immediate exceeds 32 bits")
            return Value("u32", str(value))
        if operand.startswith("c["):
            off = self.constant(operand)
            if off in (0, 4, 8):
                return self.make("u32", "mov.u32", "%ntid." + "xyz"[off // 4])
            if off == 0x28:
                return Value("stack", "")
            return self.argument(off, 4)
        require(operand in self.regs, f"unavailable 32-bit register {operand}")
        result = self.regs[operand]
        require(result.kind != "stack", "local stack pointer used")
        return result

    def cast(self, value, kind):
        if value.kind == kind and not (
            kind == "f32" and not value.name.startswith("%")
        ):
            return value
        require(
            {kind, value.kind} == {"u32", "f32"}, f"cannot use {value.kind} as {kind}"
        )
        return self.make(kind, "mov.b32", value.name)

    def write(self, register, v):
        require(
            re.fullmatch(
                r"R(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-4])", register
            ),
            f"unsupported destination {register}",
        )
        n = int(register[1:])
        for base in (n - 1, n):
            self.pairs.pop(base, None)
        self.regs[register] = v

    def pointer(self, op):
        m = re.fullmatch(r"\[R(\d+)\]", op)
        require(
            m and int(m[1]) in self.pairs,
            f"address has no tracked 64-bit register pair: {op}",
        )
        return "[" + self.pairs[int(m[1])].name + "]"

    def float_input(self, operand):
        if operand.startswith("-"):
            value = self.float_input(operand[1:])
            return self.make("f32", "neg.f32", value.name)
        return self.cast(self.read(operand), "f32")

    def step(self, raw, labels, pc):
        m = re.fullmatch(r"(?:(@!?P[0-6])\s+)?([A-Z0-9_.]+)(?:\s+(.*))?", raw)
        require(m, f"unrecognized machine instruction: {raw}")
        pred, op, operands = m.groups()
        args = tuple(part.strip() for part in operands.split(",")) if operands else ()
        self.uses[op] += 1
        if self.exited:
            if op == "BRA" and not pred and len(args) == 1:
                dest = re.fullmatch(r"`\(([^)]+)\)", args[0])
                require(
                    dest and labels.get(dest[1]) == pc, "unexpected branch after EXIT"
                )
            else:
                require(
                    op == "NOP" and not pred and not args,
                    "unexpected trailing instruction",
                )
            return
        if op == "EXIT" and not args:
            if pred:
                key = pred.lstrip("@!")
                require(
                    key in self.regs and self.regs[key].kind == "pred",
                    "undefined exit predicate",
                )
                guard = (
                    "@" + ("!" if pred.startswith("@!") else "") + self.regs[key].name
                )
                self.emit(guard + " bra", "SASS_END")
            else:
                self.exited = True
            return
        require(not pred, f"predicated {op} is not supported")
        if op == "S2R" and len(args) == 2:
            s = re.fullmatch(r"SR_(CTAID|TID)\.([XYZ])", args[1])
            require(s, f"unsupported special register {args[1]}")
            self.write(
                args[0],
                self.make("u32", "mov.u32", "%" + s[1].lower() + "." + s[2].lower()),
            )
        elif op == "MOV" and len(args) == 2:
            self.write(args[0], self.read(args[1]))
        elif op == "IMAD" and len(args) == 4:
            values = [self.cast(self.read(a), "u32").name for a in args[1:]]
            self.write(args[0], self.make("u32", "mad.lo.u32", *values))
        elif op in ("IMAD.WIDE", "IMAD.WIDE.U32") and len(args) == 4:
            base = self.argument(self.constant(args[3]), 8)
            addr = self.make("addr", "cvta.to.global.u64", base.name)
            a, b = (self.cast(self.read(v), "u32") for v in args[1:3])
            prod = self.make(
                "u64",
                "mul.wide." + ("u32" if op.endswith(".U32") else "s32"),
                a.name,
                b.name,
            )
            ptr = self.make("addr", "add.u64", addr.name, prod.name)
            dest = args[0]
            require(re.fullmatch(r"R\d+", dest), f"invalid address register {dest}")
            n = int(dest[1:])
            require(n < 254 and n % 2 == 0, f"bad register pair {dest}")
            for clobber in (n - 1, n, n + 1):
                self.pairs.pop(clobber, None)
            self.regs.pop(dest, None)
            self.regs.pop("R" + str(n + 1), None)
            self.pairs[n] = ptr
        elif op == "ISETP.GE.AND" and len(args) == 5:
            d, aux, a, b, guard = args
            require(
                re.fullmatch("P[0-6]", d) and aux == guard == "PT",
                "unsupported comparison modifiers",
            )
            values = [self.cast(self.read(v), "u32").name for v in (a, b)]
            self.regs[d] = self.make("pred", "setp.ge.s32", *values)
        elif op == "LDG.E.SYS" and len(args) == 2:
            self.write(
                args[0], self.make("u32", "ld.global.u32", self.pointer(args[1]))
            )
        elif op == "STG.E.SYS" and len(args) == 2:
            self.emit(
                "st.global.u32",
                self.pointer(args[0]),
                self.cast(self.read(args[1]), "u32").name,
            )
        elif op in ("FADD", "FMUL", "FFMA") and len(args) == (4 if op == "FFMA" else 3):
            operation = {
                "FADD": "add.rn.f32",
                "FMUL": "mul.rn.f32",
                "FFMA": "fma.rn.f32",
            }[op]
            vals = [self.float_input(v).name for v in args[1:]]
            self.write(args[0], self.make("f32", operation, *vals))
        elif op == "NOP" and not args:
            pass
        else:
            raise Unsupported(f"unsupported instruction {op} or operands {args}")

    def run(self, code, labels):
        for i, line in enumerate(code):
            try:
                self.step(line, labels, 16 * i)
            except Unsupported as e:
                raise Unsupported(f"{self.k.name}+0x{16 * i:x}: {e}") from e
        require(self.exited, f"{self.k.name}: missing terminating EXIT")
        params = ",\n".join(
            f"    .param .u{width * 8} arg{i}"
            for i, (_, width) in enumerate(self.k.params)
        )
        lines = [
            f".visible .entry {self.k.name}(\n{params}\n)",
            "{",
            *self.decl,
            *self.body,
            "SASS_END:",
            "    ret;",
            "}",
            "",
        ]
        return "\n".join(lines)


def translate(binary, nvdisasm):
    data = binary.read_bytes()
    kernels = inventory(data)
    cmd = subprocess.run(
        [nvdisasm, "-c", "-hex", str(binary.resolve())], capture_output=True, timeout=30
    )
    require(
        cmd.returncode == 0,
        "nvdisasm failed: " + cmd.stderr.decode(errors="replace").strip(),
    )
    require(binary.read_bytes() == data, "input changed during disassembly")
    code = instructions(cmd.stdout.decode(), kernels, data)
    output = [".version 7.0", ".target sm_75", ".address_size 64", ""]
    report = {
        "source": "sm_75_machine_code",
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "entries": [],
    }
    for name, kernel in kernels.items():
        lifter = Lift(kernel)
        output.append(lifter.run(*code[name]))
        report["entries"].append(
            {
                "name": name,
                "arguments": list(kernel.params),
                "opcodes": dict(sorted(lifter.uses.items())),
            }
        )
    return "\n".join(output), report


def atomic_text(path, value):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf8", dir=path.parent, delete=False
    ) as out:
        temp = out.name
        try:
            out.write(value)
            out.flush()
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


def main():
    parser = argparse.ArgumentParser(
        description="Translate standalone sm_75 CUDA binaries to PTX."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--nvdisasm", default="nvdisasm")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        paths = [p.resolve() for p in (args.input, args.output, args.report) if p]
        require(
            len(set(paths)) == len(paths), "input and output paths must be distinct"
        )
        ptx, report = translate(args.input, args.nvdisasm)
        atomic_text(args.output, ptx)
        if args.report:
            atomic_text(args.report, json.dumps(report, indent=2) + "\n")
    except (
        OSError,
        ValueError,
        UnicodeError,
        struct.error,
        subprocess.SubprocessError,
    ) as exc:
        print(f"SASS lift failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
