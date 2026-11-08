import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path


PIN = "912694289d5751ff529db94016388744084873b4"
PACKAGE = Path(__file__).resolve().parent


def git(checkout, *args, data=None, check=True):
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        input=data,
        capture_output=True,
        check=check,
    )


def digest(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def overlay_diff(checkout, entries):
    parts = []
    for name, identities in entries.items():
        local = checkout / name
        old = local.read_bytes() if local.is_file() else None
        source = (PACKAGE / "overlay" / name).read_bytes()
        if digest(source) != identities["after"]:
            raise ValueError(f"Release file differs from manifest: {name}")
        if digest(old) == identities["after"]:
            continue
        if digest(old) != identities["before"]:
            raise ValueError(f"Unexpected contents in checkout: {name}")
        old_name = "/dev/null" if old is None else "a/" + name
        parts.append(f"diff --git a/{name} b/{name}\n")
        if old is None:
            parts.append("new file mode 100644\n")
        parts.extend(difflib.unified_diff(
            (old or b"").decode("utf-8").splitlines(True),
            source.decode("utf-8").splitlines(True),
            old_name,
            "b/" + name,
            n=0,
        ))
    return "".join(parts).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Apply the PTX and Warp integration to a pinned CuMetal checkout.")
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    commit = git(checkout, "rev-parse", "HEAD").stdout.decode().strip()
    if commit != PIN:
        parser.error(f"Expected upstream commit {PIN}, got {commit}")

    entries = json.loads((PACKAGE / "overlay.json").read_text())
    build_files = json.loads((PACKAGE / "build-files.json").read_text())
    current = {name: digest((checkout / name).read_bytes()) for name in build_files}
    if all(current[name] == value["before"] for name, value in build_files.items()):
        patch = (PACKAGE / "build-integration.patch").read_bytes()
    elif all(current[name] == value["after"] for name, value in build_files.items()):
        patch = b""
    else:
        parser.error("Build files differ from the upstream and release snapshots")
    options = ["apply", "--unidiff-zero", "--whitespace=nowarn"]

    patch += overlay_diff(checkout, entries)
    if not patch:
        print("The release sources are already applied.")
        return
    flags = ["--check"] if args.check else []
    result = git(checkout, *options, *flags, data=patch, check=False)
    if result.returncode:
        parser.error(result.stderr.decode())
    print("Patch verified." if args.check else "PTX and Warp integration applied.")


if __name__ == "__main__":
    main()

