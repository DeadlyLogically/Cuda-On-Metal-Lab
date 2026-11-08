#!/usr/bin/env python3

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

FUNCTIONAL = Path(__file__).resolve().parents[1] / "functional"
sys.path.insert(0, str(FUNCTIONAL))
import run_warp_sweep as sweep

class SweepEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.identifier = "warp.tests.test_math.TestMath.test_scalar_math_cuda_0"
        self.checks = {
            "passed": True, "test": self.identifier, "tests_run": 1,
            "failures": 0, "errors": 0, "skipped": 0,
            "expected_failures": 0, "unexpected_successes": 0,
            "source_unchanged": True, "output_config": None,
            "device": "cuda:0", "device_name": "Apple M2", "selected_output": "ptx",
        }
        self.record = {
            "event": "kernel_launch", "source": "generic_ptx",
            "provenance": "generic_ptx_lowering", "device": "apple_gpu",
            "device_name": "Apple M2", "launch_success": "true", "semantic_quality": "exact",
        }

    def validate(self, checks=None, records=None, code=0, timeout=False):
        return sweep.validate_result(
            self.checks if checks is None else checks,
            [self.record] if records is None else records,
            code, timeout, "cuda:0", self.identifier,
        )

    def test_exact_gpu_and_software_fp64_evidence(self):
        self.assertFalse(self.validate())
        self.record["provenance"] = "generic_ptx_lowering_fp64_ieee64"
        self.assertFalse(self.validate())

    def test_incomplete_or_skipped_numerics_fail(self):
        for key, value in (
            ("passed", False), ("test", "different.test"), ("tests_run", 0),
            ("tests_run", 2), ("failures", 1), ("errors", 1), ("skipped", 1),
            ("expected_failures", 1), ("unexpected_successes", 1),
            ("source_unchanged", False), ("output_config", "ptx"),
            ("device", "cpu"), ("selected_output", "cubin"),
        ):
            with self.subTest(key=key, value=value):
                check = {**self.checks, key: value}
                self.assertTrue(self.validate(checks=check))
        self.assertTrue(self.validate(checks={}))

    def test_unexpected_trace_or_missing_trace_fails(self):
        self.assertTrue(self.validate(records=[]))
        for key, value in (
            ("event", "module_load"), ("source", "workload_specialization"),
            ("provenance", "generic_ptx_lowering_fp64_fast48"), ("device", "cpu"),
            ("device_name", "Apple M4"), ("launch_success", "false"),
            ("semantic_quality", "approximate"),
        ):
            with self.subTest(key=key):
                self.assertTrue(self.validate(records=[self.record, {**self.record, key: value}]))

    def test_success_report_cannot_hide_crash_or_timeout(self):
        self.assertTrue(self.validate(code=1))
        self.assertTrue(self.validate(code=-9))
        self.assertTrue(self.validate(timeout=True))

    def test_cpu_baseline_cannot_claim_gpu_work(self):
        identifier = self.identifier.removesuffix("cuda_0") + "cpu"
        checks = {**self.checks, "test": identifier, "device": "cpu", "device_name": "arm", "selected_output": None}
        self.assertFalse(sweep.validate_result(checks, [], 0, False, "cpu", identifier))
        self.assertTrue(sweep.validate_result(checks, [self.record], 0, False, "cpu", identifier))

    def test_provenance_shared_with_unittest_progress_is_retained(self):
        line = 'test_name ... CUMETAL_PROVENANCE event=kernel_launch source=generic_ptx device_name="Apple M2"\n'
        self.assertEqual(sweep.provenance(line), [{"event": "kernel_launch", "source": "generic_ptx", "device_name": "Apple M2"}])

    def test_fixed_manifest_and_rejection_of_invalid_selection(self):
        path = FUNCTIONAL / "warp_sweep_manifest.json"
        manifest = sweep.load_manifest(path)
        self.assertEqual(len(manifest["cases"]), 24)
        for mutate in (
            lambda obj: obj.update(warp_commit="wrong"),
            lambda obj: obj["cases"].append(obj["cases"][0]),
            lambda obj: obj["cases"][0].update(case="../bad"),
            lambda obj: obj["cases"][0].update(test="warp.tests.bad.test_cpu"),
            lambda obj: obj.update(cases=[]),
        ):
            candidate = copy.deepcopy(manifest)
            mutate(candidate)
            with tempfile.TemporaryDirectory() as directory:
                invalid = Path(directory) / "invalid.json"
                invalid.write_text(json.dumps(candidate))
                with self.assertRaises(ValueError):
                    sweep.load_manifest(invalid)

    def test_individual_passes_cannot_hide_rebuild_or_missing_tests(self):
        identity = {"compiler_sha256": "compiler-a", "runtime_sha256": "runtime-a",
                    "launcher_sha256": "launcher-a", "warp_native_sha256": "native-a"}
        report = {"identity": identity, "selected_total": 2,
                  "cases": [{"passed": True}, {"passed": True}]}
        sweep.finalize_report(report, dict(identity))
        self.assertTrue(report["passed"])
        self.assertTrue(report["artifacts_unchanged"])
        for key in identity:
            candidate = copy.deepcopy(report)
            sweep.finalize_report(candidate, {**identity, key: "rebuilt"})
            self.assertFalse(candidate["passed"])
            self.assertFalse(candidate["artifacts_unchanged"])
            self.assertTrue(candidate["run_errors"])
        incomplete = copy.deepcopy(report)
        incomplete["cases"].pop()
        sweep.finalize_report(incomplete, dict(identity))
        self.assertFalse(incomplete["passed"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
