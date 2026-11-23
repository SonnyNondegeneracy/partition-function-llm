"""Fetch an explicitly pinned base model (not part of CPU reproduction)."""
from pathlib import Path
import argparse,json

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--size',choices=['9B','4B'],default='9B');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 from huggingface_hub import snapshot_download
 repo='Qwen/Qwen3.5-'+a.size;revision={'9B':'c202236235762e1c871ad0ccb60c8ee5ba337b9a','4B':'851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'}[a.size]
 snapshot_download(repo_id=repo,revision=revision,local_dir=str(a.output))
 (a.output/'model_manifest.json').write_text(json.dumps(dict(id=repo,revision=revision),indent=2)+'\n')
 print(str(a.output.resolve()))
if __name__=='__main__':main()
