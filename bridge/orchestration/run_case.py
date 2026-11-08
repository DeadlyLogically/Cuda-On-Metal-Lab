import argparse
import hashlib
import json
import os
import selectors
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
CUDART = Path("libcudart.so.13.0.96")
EXPECTED_CUDART = "7bdba2b5b08cbdc85203c41cc94598adedb1bcfea7cb574ca693ac73599e4e63"
MAX_LINE = 1024 * 1024

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(mode, report_dir):
    host = (report_dir/"host.log").read_text(errors="replace").splitlines()
    guest = (report_dir/"guest.log").read_text(errors="replace")
    wire = [json.loads(row) for row in (report_dir/"wire.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in host if line.startswith('{"session":')]
    traces = [line for line in host if line.startswith("CUMETAL_PROVENANCE ") and all(
        field in line.split() for field in ('event=kernel_launch','kernel="bridge_add"',
        'source=generic_ptx','provenance=generic_ptx_lowering','semantic_quality=exact',
        'device=apple_gpu','launch_success=true')
    )]
    digests = [hashlib.sha256((entry['request']+'\n').encode()).hexdigest() for entry in wire]
    checks = {"matched_host_requests": bool(wire) and digests == [event["request_sha256"] for event in events]}
    if mode != "driver":
        checks["original_runtime_function_resolved"] = "cudaMalloc-source=/original/libcudart.so.13.0.96" in guest
        checks["original_copy_roundtrip"] = ("original-runtime-copy-verified" in guest and
            any(row["request"].startswith("D2H ") and row["reply"]=="OK 010000000200000003000000" for row in wire) and
            any(row["request"].startswith("FREE ") and row["reply"]=="OK" for row in wire))
    if mode != "runtime-mem":
        checks.update(exact_gpu_dispatch=len(traces)==1,
                      linux_verified_result="value=[5,7,9]" in guest.splitlines(),
                      successful_launch=any(row["request"].startswith("LAUNCH ") and row["reply"]=="OK" for row in wire),
                      synchronized=any(row["request"]=="SYNC" and row["reply"]=="OK" for row in wire),
                      correct_readback=any(row["request"].startswith("D2H ") and row["reply"]=="OK 050000000700000009000000" for row in wire))
    return checks, traces

def run(mode, report_dir, image, guest_source, runtime, library, private):
    if digest(runtime) != EXPECTED_CUDART:
        raise ValueError("libcudart input SHA-256 does not match the official inspected file")
    report_dir.mkdir(exist_ok=False, parents=True)


    snapshot=(report_dir/"tested-source").resolve()
    source=snapshot/"guest"
    source.mkdir(parents=True)
    for path in guest_source.iterdir():
        if path.is_file() and path.suffix in {".c", ".h", ".py", ".sh"}:
            shutil.copyfile(path,source/path.name)
    for path in [LAB/"host/bridge_host.py", Path(__file__).with_name("guest_tunnel.py")]:
        shutil.copyfile(path,snapshot/path.name)
    if private:
        for file in ("private_exports.c", "bridge_md2.h"):
            shutil.copyfile(LAB/"analysis"/file,snapshot/file)
    identities={str(path.relative_to(snapshot)):digest(path) for path in snapshot.rglob("*") if path.is_file()}
    (report_dir/"source-sha256.json").write_text(json.dumps(identities,indent=2)+"\n")
    name = "cumetal-pilot-" + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="cm-bridge-", dir="/tmp") as td:

        sockpath = Path(td) / "h.sock"
        with (report_dir/"host.log").open("wb") as host_log, (report_dir/"guest.log").open("wb") as guest_log, (report_dir/"wire.jsonl").open("w",buffering=1) as wire:
            host = subprocess.Popen([sys.executable,str(snapshot/"bridge_host.py"),"--library",str(library),"--socket",str(sockpath)], stdout=host_log,stderr=subprocess.STDOUT)
            guest = None
            deadline = time.monotonic()+100
            commands = []
            error = None
            try:
                while not sockpath.exists():
                    if host.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError("host service did not create a socket")
                    time.sleep(.05)
                private_options=[]
                if private:
                    private_options=["-v",f"{snapshot/'private_exports.c'}:/abi/private_exports.c:ro","-v",f"{snapshot/'bridge_md2.h'}:/abi/bridge_md2.h:ro","-e","BRIDGE_PRIVATE_EXPORTS=/abi/private_exports.c"]
                command = ["docker","run","--rm","-i","--platform","linux/arm64","--network","none","--name",name,
                           "-v", f"{source}:/src:ro","-v", f"{snapshot/'guest_tunnel.py'}:/lab/guest_tunnel.py:ro",
                           "-v", f"{runtime}:/original/libcudart.so.13.0.96:ro",*private_options,image,"/bin/sh","-ec",
                           'cd /src; sh ./build.sh /tmp/guest-build >&2; export LD_LIBRARY_PATH=/tmp/guest-build; exec python3 /lab/guest_tunnel.py /tmp/guest-build/probe "$@"',
                           "probe",mode]
                if mode != "driver":
                    command.append("/original/libcudart.so.13.0.96")
                guest=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=guest_log,bufsize=0)
                with selectors.DefaultSelector() as selector, socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as backend:
                    selector.register(guest.stdout,selectors.EVENT_READ)
                    backend.connect(str(sockpath))
                    backend.settimeout(30)
                    with backend.makefile('rb') as answers:
                        while True:
                            remaining=deadline-time.monotonic()
                            if remaining<=0:raise TimeoutError("guest run deadline")
                            if not selector.select(remaining):raise TimeoutError("guest relay deadline")
                            request=guest.stdout.readline(MAX_LINE+1)
                            if not request:break
                            if len(request)>MAX_LINE or not request.endswith(b"\n"):
                                raise ValueError("malformed guest wire line")
                            op=request.split(maxsplit=1)[0].decode('ascii')
                            commands.append(op)
                            backend.sendall(request)
                            reply=answers.readline(MAX_LINE+1)
                            if len(reply)>MAX_LINE or not reply.endswith(b"\n"):
                                raise ValueError("malformed host response")
                            wire.write(json.dumps({"request":request.decode().strip(),"reply":reply.decode().strip()})+"\n")
                            guest.stdin.write(reply)
                exitcode=guest.wait(timeout=10)
            except (OSError,EOFError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
                error=str(exc)
                exitcode=1
            finally:
                if guest and guest.poll() is None:
                    subprocess.run(["docker","rm","-f",name],capture_output=True,timeout=15)
                    guest.kill()
                    guest.wait()
                host.terminate()
                try:host.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    host.kill()
                    host.wait()
    checks,gpu_lines=verify(mode,report_dir)
    passed=exitcode==0 and all(checks.values())
    result=dict(mode=mode,experimental_private_tables=private,guest_exit=exitcode,error=error,passed=passed,checks=checks,commands=commands,generic_gpu_launches=len(gpu_lines),gpu_provenance=gpu_lines,libcudart_sha256=digest(runtime),original_preserved=digest(runtime)==EXPECTED_CUDART,host_cumetal_sha256=digest(library),image=image)
    (report_dir/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False))
    return 0 if passed else (exitcode or 1)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["driver","runtime","runtime-mem"])
    parser.add_argument("output",type=Path)
    parser.add_argument("--image",default="cumetal-bridge-lab:bookworm")
    parser.add_argument("--private",action="store_true",help="select experimental export tables for pinned runtime")
    parser.add_argument("--guest",type=Path,default=LAB/"guest")
    parser.add_argument("--cudart",type=Path,default=CUDART)
    parser.add_argument("--cumetal",type=Path,required=True,help="path to the CuMetal driver library built on this Mac")
    args=parser.parse_args()
    sys.exit(run(args.mode,args.output,args.image,args.guest.resolve(),args.cudart.resolve(),args.cumetal.resolve(),args.private))
