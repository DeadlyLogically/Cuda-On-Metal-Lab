import argparse
import re
import subprocess
from pathlib import Path

PIN = "912694289d5751ff529db94016388744084873b4"
TEMPLATE = Path(__file__).with_name("prerequisites.patch.in")


def materialize(checkout):
    result = []
    upstream = []
    for line in TEMPLATE.read_text().splitlines(keepends=True):
        if line.startswith("--- a/"):
            relative = line.rstrip()[6:]
            upstream = (
                subprocess.check_output(
                    ["git", "-C", str(checkout), "show", f"{PIN}:{relative}"]
                )
                .decode()
                .splitlines(keepends=True)
            )
        elif line.startswith("--- /dev/null"):
            upstream = []
        marker = re.fullmatch(r"-@UPSTREAM_LINE=(\d+)@\n", line)
        if marker:
            line = "-" + upstream[int(marker.group(1)) - 1]
        result.append(line)
    return "".join(result)


def main():
    parser = argparse.ArgumentParser(
        description="Apply prerequisites to a clean pinned CuMetal checkout."
    )
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    git = ["git", "-C", str(args.checkout.resolve())]
    head = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if head != PIN:
        parser.error(f"Expected commit {PIN}, found {head}")
    if subprocess.run(git + ["diff", "--quiet", "HEAD"]).returncode != 0:
        parser.error("Use a clean checkout of the pinned commit.")
    patch = materialize(args.checkout.resolve())
    subprocess.run(
        git + ["apply", "--check", "--unidiff-zero", "-"],
        input=patch,
        text=True,
        check=True,
    )
    subprocess.run(
        git + ["apply", "--unidiff-zero", "-"], input=patch, text=True, check=True
    )


if __name__ == "__main__":
    main()
