"""Load base + one cumulative adapter at its fixed base-model revision."""
from pathlib import Path
import argparse,json,hashlib

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--adapter',default='main_9b/T1/round_08');p.add_argument('--check-only',action='store_true');a=p.parse_args()
 row=next(r for r in json.loads((a.data_root/'metadata/adapters.json').read_text()) if r['id']==a.adapter);folder=a.data_root/row['path']
 assert hashlib.sha256((folder/'adapter_model.safetensors').read_bytes()).hexdigest()==row['weights_sha256']
 if a.check_only:print(json.dumps(row,indent=2));return
 from transformers import AutoModelForImageTextToText,AutoTokenizer
 from peft import PeftModel
 import torch
 tokenizer=AutoTokenizer.from_pretrained(row['base_model'],revision=row['base_revision'])
 base=AutoModelForImageTextToText.from_pretrained(row['base_model'],revision=row['base_revision'],torch_dtype=torch.bfloat16,device_map='auto')
 model=PeftModel.from_pretrained(base,str(folder)).eval()
 print(type(model).__name__,type(tokenizer).__name__)
if __name__=='__main__':main()
