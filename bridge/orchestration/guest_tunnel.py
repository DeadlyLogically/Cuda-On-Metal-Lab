import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_LINE = 1024 * 1024

def run(command):
    with tempfile.TemporaryDirectory(prefix="cuda-wire-") as directory:
        path = str(Path(directory) / "driver.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(path)
            listener.listen(1)
            listener.settimeout(0.15)
            env = dict(os.environ, CUMETAL_BRIDGE_SOCKET=path)
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=sys.stderr)
            try:
                while child.poll() is None:
                    try:
                        connection, _ = listener.accept()
                    except TimeoutError:
                        continue
                    with connection, connection.makefile("rb") as incoming:
                        while request := incoming.readline(MAX_LINE + 1):
                            if len(request) > MAX_LINE or not request.endswith(b"\n"):
                                raise ValueError("invalid guest message")
                            sys.stdout.buffer.write(request)
                            sys.stdout.buffer.flush()
                            reply = sys.stdin.buffer.readline(MAX_LINE + 1)
                            if len(reply) > MAX_LINE or not reply.endswith(b"\n"):
                                raise EOFError("host relay closed")
                            connection.sendall(reply)
                return child.returncode if child.returncode >= 0 else 128 - child.returncode
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait()

if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
