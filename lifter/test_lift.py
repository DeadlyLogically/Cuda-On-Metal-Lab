import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from lift import Unsupported, inventory, instructions, Lift


HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "fixtures" / "build"
DISASSEMBLER = os.getenv(
    "CUMETAL_SASS_NVDISASM", str(HERE.parent / "mac" / "nvdisasm-docker")
)


def text_extent(blob, section):
    head = struct.unpack_from("<16sHHIQQQIHHHHHH", blob)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", blob, head[6] + i * 64)
        for i in range(head[12])
    ]
    strings = sections[head[13]]
    table = blob[strings[4] : strings[4] + strings[5]]
    for entry in sections:
        if table[entry[0] :].split(b"\0", 1)[0] == section.encode():
            return entry[4], entry[5]
    raise AssertionError(section)


class BinaryTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.dir = Path(self.scratch.name)
        self.out = self.dir / "lift.ptx"
        self.report = self.dir / "report.json"

    def invoke(self, f):
        return subprocess.run(
            [
                sys.executable,
                str(HERE / "lift.py"),
                str(f),
                "-o",
                str(self.out),
                "--nvdisasm",
                DISASSEMBLER,
                "--report",
                str(self.report),
            ],
            capture_output=True,
            text=True,
        )

    def test_opcode_bytes_change_output_with_same_name_and_arguments(self):
        add = bytearray((FIXTURES / "vectorAdd.sm75.cubin").read_bytes())
        mul = (FIXTURES / "elementMultiply.sm75.cubin").read_bytes()
        original = self.invoke(FIXTURES / "vectorAdd.sm75.cubin")
        self.assertEqual(original.returncode, 0, original.stderr)
        self.assertIn("add.rn.f32", self.out.read_text())
        a, az = text_extent(add, ".text.vectorAdd")
        m, mz = text_extent(mul, ".text.elementMultiply")
        self.assertEqual(az, mz)
        changed = [
            i
            for i in range(0, az, 16)
            if add[a + i : a + i + 16] != mul[m + i : m + i + 16]
        ]
        self.assertEqual(len(changed), 1)
        pc = changed[0]
        add[a + pc : a + pc + 16] = mul[m + pc : m + pc + 16]
        candidate = self.dir / "different.cubin"
        candidate.write_bytes(add)
        result = self.invoke(candidate)
        self.assertEqual(result.returncode, 0, result.stderr)
        lifted = self.out.read_text()
        self.assertIn(".entry vectorAdd(", lifted)
        self.assertIn("mul.rn.f32", lifted)
        self.assertNotIn("add.rn.f32", lifted)
        self.assertEqual(
            json.loads(self.report.read_text())["entries"][0]["opcodes"]["FMUL"], 1
        )

    def test_unknown_atomic_reports_exact_offset_and_emits_no_ptx(self):
        result = self.invoke(FIXTURES / "atomicSum.sm75.cubin")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "atomicSum+0xa0: unsupported instruction RED.E.ADD", result.stderr
        )
        self.assertFalse(self.out.exists())
        self.assertFalse(self.report.exists())

    def test_wrong_gpu_truncated_binary_and_incomplete_listing(self):
        data = (FIXTURES / "vectorAdd.sm75.cubin").read_bytes()
        kernels = inventory(data)
        for bad in [data[:120], data[:48] + bytes([80]) + data[49:]]:
            with self.assertRaises(Unsupported):
                inventory(bad)
        listing = (FIXTURES / "vectorAdd.sm75.hex.sass").read_text()
        with self.assertRaisesRegex(Unsupported, "noncontiguous"):
            instructions(listing.replace("/*0030*/", ""), kernels, data)
        code, labels = instructions(listing, kernels, data)["vectorAdd"]
        code[3] = "BRA `(.L_x_0)"
        with self.assertRaisesRegex(Unsupported, "unsupported instruction BRA"):
            Lift(kernels["vectorAdd"]).run(code, labels)


if __name__ == "__main__":
    unittest.main()
