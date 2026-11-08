import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
ARCHIVE_HASH = "e2c78564330a09d890f98a066a2082a32c78256ffcc6bccca1696bd2abbaf445"
RUNTIME_HASH = "7bdba2b5b08cbdc85203c41cc94598adedb1bcfea7cb574ca693ac73599e4e63"

def main():
    with urllib.request.urlopen(BASE+"redistrib_13.0.2.json",timeout=20) as resp:
        entry=json.load(resp)["cuda_cudart"]["linux-sbsa"]
    if entry["sha256"] != ARCHIVE_HASH:
        raise ValueError("Unexpected NVIDIA manifest archive digest")
    with urllib.request.urlopen(BASE+entry["relative_path"],timeout=30) as resp:
        data=resp.read()
    if hashlib.sha256(data).hexdigest()!=ARCHIVE_HASH:raise ValueError("Archive digest mismatch")
    with tarfile.open(fileobj=io.BytesIO(data),mode="r:xz") as archive:
        member=next(entry for entry in archive if entry.isfile() and entry.name.endswith("/lib/libcudart.so.13.0.96"))
        lib=archive.extractfile(member).read()
    if hashlib.sha256(lib).hexdigest()!=RUNTIME_HASH:raise ValueError("Binary digest mismatch")
    dest=Path("libcudart.so.13.0.96")
    if dest.exists():
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=RUNTIME_HASH:
            raise FileExistsError(str(dest))
    else:
        with dest.open("xb") as out:out.write(lib)
    print(dest.resolve(),RUNTIME_HASH)
if __name__=="__main__":
    main()
