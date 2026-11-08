#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

PINS = {
    "warp": "e6c3ba2d54bb048115760b5cd7a4bb2573329ae7",
    "cub": "c493b3bf143ccd66c917a1982ea64bb2a8a82932",
    "thrust": "1ac51f2b6219ff17d15d93f2e0be85038556f346",
}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_commands(build_log):
    compile_commands, link_commands = [], []
    for line in build_log.read_text().splitlines():
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if not tokens or line.startswith("Output from:"):
            continue
        if Path(tokens[0]).name == "nvcc" and "-c" in tokens and "-o" in tokens:
            if any(Path(token).name == "reduce.cu" for token in tokens):
                compile_commands.append(tokens)
        if Path(tokens[0]).name in ("g++", "clang++", "c++") and "-shared" in tokens and "-o" in tokens:
            if Path(tokens[tokens.index("-o") + 1]).name == "libwarp.dylib":
                link_commands.append(tokens)
    require(len(compile_commands) == 1 and len(link_commands) == 1,
            "build log must contain exactly one reduce.cu compile and one libwarp link command")
    return compile_commands[0], link_commands[0]

def prepare_commands(build_log, prepared, variant, cub, thrust, nvcc, output):
    compile_command, link_command = read_commands(build_log)
    logged_source = next(Path(token) for token in compile_command if Path(token).name == "reduce.cu")
    logged_native = logged_source.parent
    require(logged_native.name == "native", "unexpected native source layout in build log")
    compile_command[0] = str(nvcc)
    for index, argument in enumerate(compile_command):
        if argument == "-I" + str(logged_native):
            compile_command[index] = "-I" + str(variant / "warp/native")
        elif argument == str(logged_source):
            compile_command[index] = str(variant / "warp/native/reduce.cu")
    object_path = output / "reduce-upstream.cu.o"
    compile_command[compile_command.index("-o") + 1] = str(object_path)
    compile_command[1:1] = ["-I", str(cub), "-I", str(thrust),
                            "-Wno-missing-template-arg-list-after-template-kw"]
    replaced = 0
    for index, argument in enumerate(link_command):
        path = Path(argument)
        if path.suffix != ".o":
            continue
        require(path.parent == logged_native, "link log contains an unexpected object directory")
        if path.name == "reduce.cu.o":
            link_command[index] = str(object_path)
            replaced += 1
        else:
            link_command[index] = str(prepared / "warp/native" / path.name)
    require(replaced == 1, "expected exactly one reduce.cu.o input in the link command")
    binary = variant / "warp/bin/libwarp.dylib"
    link_command[link_command.index("-o") + 1] = str(binary)
    objects = [Path(token) for token in link_command if token.endswith(".o") and token != str(object_path)]
    require(len(objects) == 24 and len(set(objects)) == 24,
            "expected the 24 unchanged native objects from the pinned prepared build")
    return compile_command, link_command, object_path, binary, objects

def check_pin(path, kind, clean=False):
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    require(commit == PINS[kind], f"expected {kind} {PINS[kind]}, found {commit}")
    if clean:
        status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
        require(not status, f"{kind} checkout must be unchanged")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-warp", "variant-warp", "cub", "thrust", "build-log", "nvcc", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    prepared, variant, cub, thrust, build_log, nvcc, output = (
        getattr(args, name).resolve() for name in
        ("prepared_warp", "variant_warp", "cub", "thrust", "build_log", "nvcc", "output"))
    try:
        for source in (prepared, cub, thrust):
            for destination in (variant, output):
                require(destination != source and source not in destination.parents and destination not in source.parents,
                        "variant/output must not overlap any source checkout")
        require(variant != output and variant not in output.parents and output not in variant.parents,
                "variant and output directories must be separate")
        require(not variant.exists(), "variant directory must be new")
        require(not output.exists(), "output directory must be new to preserve earlier evidence")
        require(nvcc.is_file() and os.access(nvcc, os.X_OK), "nvcc shim must be an existing executable")
        check_pin(prepared, "warp")
        check_pin(cub, "cub", clean=True)
        check_pin(thrust, "thrust", clean=True)
        original = subprocess.check_output(["git", "-C", str(prepared), "show", "HEAD:warp/native/reduce.cu"])
        require((prepared / "warp/native/reduce.cu").read_bytes() == original,
                "prepared Warp reduction source differs from upstream")
        compile_command, link_command, object_path, binary, objects = prepare_commands(
            build_log, prepared, variant, cub, thrust, nvcc, output)
        before = {str(path): sha256(path) for path in objects}
        original_binary_hash = sha256(prepared / "warp/bin/libwarp.dylib")
        cpu_compiler_hash = sha256(prepared / "warp/bin/libwarp-clang.dylib")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    output.mkdir(parents=True)
    variant.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--shared",
                    str(prepared), str(variant)], check=True)
    patches = Path(__file__).resolve().parent / "warp-patches/apply_warp_patches.sh"
    with (output / "patches.log").open("w") as sink:
        subprocess.run(["bash", str(patches), str(variant)], stdout=sink, stderr=subprocess.STDOUT, check=True)
    source = variant / "warp/native/reduce.cu"
    require(source.read_bytes() == original, "patching changed Warp's reduction source")
    binary.parent.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "pins": PINS, "reduce_source_sha256": sha256(source),
              "reduce_source_unchanged": True, "original_binary_sha256": original_binary_hash,
              "unchanged_objects": before, "compile_command": compile_command, "link_command": link_command}
    (output / "commands.json").write_text(json.dumps(report, indent=2) + "\n")
    env = dict(os.environ)
    if not env.get("SDKROOT"):
        env["SDKROOT"] = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    env["TMPDIR"] = str(output)
    for phase, command in (("compile", compile_command), ("link", link_command)):
        start = time.monotonic()
        with (output / (phase + ".log")).open("w") as sink:
            result = subprocess.run(command, env=env, cwd=variant, stdout=sink, stderr=subprocess.STDOUT, timeout=300)
        report[phase] = {"returncode": result.returncode, "elapsed_seconds": time.monotonic() - start,
                         "log": str(output / (phase + ".log"))}
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{phase}: {result.returncode}", flush=True)
        if result.returncode:
            return result.returncode
    require(before == {str(path): sha256(path) for path in objects}, "an original non-reduction object changed")
    require(sha256(prepared / "warp/bin/libwarp.dylib") == original_binary_hash,
            "original prepared Warp binary changed")
    require(source.read_bytes() == original, "reduction source changed during build")
    shutil.copy2(prepared / "warp/bin/libwarp-clang.dylib", binary.with_name("libwarp-clang.dylib"))
    report.update(passed=True, binary=str(binary), binary_sha256=sha256(binary),
                  original_binary_preserved=True, original_objects_preserved=True,
                  reduction_object_sha256=sha256(object_path), cpu_compiler_sha256=cpu_compiler_hash)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "binary": str(binary), "report": str(output / "result.json")}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
