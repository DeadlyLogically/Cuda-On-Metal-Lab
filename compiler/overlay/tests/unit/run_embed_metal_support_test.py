#!/usr/bin/env python3

import argparse
from pathlib import Path
import subprocess
import tempfile
import time

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def cmake_literal(value):
    value = str(value)
    delimiter = "="
    while "]" + delimiter + "]" in value:
        delimiter += "="
    return "[" + delimiter + "[" + value + "]" + delimiter + "]"

def run(command, expect_success=True):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=30)
    if (result.returncode == 0) != expect_success:
        raise RuntimeError(f"unexpected status {result.returncode}: {command}\n{result.stdout}")
    return result.stdout

def write_project(directory, module):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.28)\n"
        "project(MetalSupportFixture LANGUAGES NONE)\n"
        f"include({cmake_literal(module)})\n"
        'cumetal_expand_metal_support("${CMAKE_CURRENT_SOURCE_DIR}/root.metal" expanded)\n'
        'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/expanded.metal" "${expanded}")\n'
        'get_property(dependencies DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS)\n'
        'list(JOIN dependencies "\\n" dependency_lines)\n'
        'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dependencies.txt" "${dependency_lines}\\n")\n'
    )

def configure(args, source, build, expect_success=True):
    command = [str(args.cmake), "-S", str(source), "-B", str(build),
               "-G", args.generator]
    if args.make_program:
        command.append(f"-DCMAKE_MAKE_PROGRAM={args.make_program}")
    return run(command, expect_success)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True, type=Path)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--make-program", default="")
    parser.add_argument("--work-root", required=True, type=Path)
    args = parser.parse_args()
    args.cmake = args.cmake.resolve()
    args.module = args.module.resolve()
    args.work_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="embed-support-", dir=args.work_root) as temporary:
        root = Path(temporary)
        source = root / "ordered"
        build = root / "ordered-build"
        write_project(source, args.module)
        (source / "nested").mkdir()
        (source / "root.metal").write_text(
            '#define ROOT_BEGIN 1\n#include "nested/middle.metal"\n#define ROOT_END 1\n')
        (source / "nested/middle.metal").write_text(
            '#define MIDDLE_BEGIN 1\n#include "leaf.metal"\n#define MIDDLE_END 1\n')
        leaf = source / "nested/leaf.metal"
        leaf.write_text("#include <metal_stdlib>\n#define LEAF_ORIGINAL 1\n")
        configure(args, source, build)
        expanded = (build / "expanded.metal").read_text()
        markers = ["ROOT_BEGIN", "MIDDLE_BEGIN", "LEAF_ORIGINAL", "MIDDLE_END", "ROOT_END"]
        positions = [expanded.index(marker) for marker in markers]
        require(positions == sorted(positions), "nested includes lost source order")
        require('#include "' not in expanded, "literal quoted include was not expanded")
        require("#include <metal_stdlib>" in expanded, "system include was not preserved")
        actual_dependencies = set((build / "dependencies.txt").read_text().splitlines())
        expected_dependencies = {str(path.resolve()) for path in (
            source / "root.metal", source / "nested/middle.metal", leaf)}
        require(actual_dependencies == expected_dependencies,
                f"configure dependencies differ: {actual_dependencies} != {expected_dependencies}")

        time.sleep(1.1)
        leaf.write_text("#include <metal_stdlib>\n#define LEAF_UPDATED 1\n")
        run([str(args.cmake), "--build", str(build)])
        rebuilt = (build / "expanded.metal").read_text()
        require("LEAF_UPDATED" in rebuilt and "LEAF_ORIGINAL" not in rebuilt,
                "editing an included leaf did not trigger CMake regeneration")

        missing = root / "missing"
        write_project(missing, args.module)
        (missing / "root.metal").write_text('#include "absent.metal"\n')
        log = configure(args, missing, root / "missing-build", False)
        require("Missing bundled Metal support" in log, "missing include failed for wrong reason")

        cycle = root / "cycle"
        write_project(cycle, args.module)
        (cycle / "root.metal").write_text('#include "child.metal"\n')
        (cycle / "child.metal").write_text('#include "root.metal"\n')
        log = configure(args, cycle, root / "cycle-build", False)
        require("Cyclic bundled Metal include" in log, "include cycle failed for wrong reason")

        for name, directive in (
            ("unterminated", '#include "unclosed.metal\n'),
            ("macro", "#include SUPPORT_PATH\n"),
        ):
            source = root / name
            write_project(source, args.module)
            (source / "root.metal").write_text(directive)
            log = configure(args, source, root / (name + "-build"), False)
            require("Unsupported bundled Metal include" in log,
                    f"{name} include failed for wrong reason: {log}")

    print("PASS: bundled Metal includes preserve order/system includes, track all dependencies, "
          "regenerate after leaf edits, and reject missing/cyclic/malformed/macro includes")

if __name__ == "__main__":
    main()
