from pathlib import Path
import hashlib,json
def writable(path): return Path(path).expanduser().resolve()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,data):
    p=writable(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
