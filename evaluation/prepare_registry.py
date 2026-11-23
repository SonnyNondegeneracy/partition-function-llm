"""Create local path registries; never change the archived contracts."""
from pathlib import Path
import argparse,json,hashlib
CODE=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--run',default='main_T1');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 data=a.data_root.resolve();out=a.output.resolve();model=str(a.model.resolve());cfg=json.loads((CODE/'configs/training_runs.json').read_text())[a.run]
 if out.is_relative_to(data) or out.is_relative_to(CODE):p.error('Output must be outside the archives.')
 out.mkdir(parents=True,exist_ok=False)
 group='main_9b/'+cfg['task'] if a.run.startswith('main_') else 'model_4b/T1' if a.run=='model_4b' else 'kl_scan/'+a.run
 policies={'base':dict(round=0,adapter=None,adapter_sha256=None,training_task=None)}
 for rd in range(1,9):
  ad=data/f'adapters/{group}/round_{rd:02}';policies[f'{cfg["task"]}_r{rd:02}']=dict(round=rd,training_task=cfg['task'],adapter=str(ad),adapter_sha256=hashlib.sha256((ad/'adapter_model.safetensors').read_bytes()).hexdigest(),adapter_config_sha256=hashlib.sha256((ad/'adapter_config.json').read_bytes()).hexdigest())
 (out/'policies.json').write_text(json.dumps(dict(model=model,policies=policies),indent=2)+'\n')
 registry=json.loads((data/'metadata/supplementary/prepared/registry.json').read_text());registry['model']=model
 for task,item in registry['tasks'].items():
  item['spec']=str(data/'metadata/supplementary/prepared/specs'/Path(item['spec']).name)
  for mode,entry in item['prepared'].items():entry['path']=str(data/'metadata/supplementary/prepared'/Path(entry['path']).name)
 if a.run=='model_4b':
  prepared4=data/cfg['prepared']
  registry['tasks']['T1']['prepared']['direct'].update(path=str(prepared4),sha256=hashlib.sha256(prepared4.read_bytes()).hexdigest())
 (out/'tasks.json').write_text(json.dumps(registry,indent=2)+'\n')
 prepared=data/cfg['prepared'];extension=dict(model=model,task=dict(name='T1',prepared=str(prepared),prepared_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest()),policies=policies)
 config=Path(model)/'config.json'
 if config.is_file():extension['model_config_sha256']=hashlib.sha256(config.read_bytes()).hexdigest()
 (out/'extensions.json').write_text(json.dumps(extension,indent=2)+'\n')
 if a.run.startswith('main_'):
  formal=json.loads((data/'metadata/main/config/measurement_manifest.json').read_text());formal['model']=model
  for policy,entry in formal['policies'].items():
   if policy=='base':continue
   task=policy.split('_')[0];rd=int(policy[-2:]);entry['adapter']=str(data/f'adapters/main_9b/{task}/round_{rd:02}')
  (out/'formal_contract.json').write_text(json.dumps(formal,indent=2)+'\n')
 print(json.dumps(dict(run=a.run,model=cfg['model'],revision=cfg['revision'],output=str(out),model_config_available=config.exists())))
if __name__=='__main__':main()
