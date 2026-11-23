"""Expand archived training invocations; execute only with --execute."""
from pathlib import Path
import argparse,json,os,sys,subprocess,hashlib,shlex,re,ast,math
CODE=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data-root',type=Path,required=True);p.add_argument('--run',required=True);p.add_argument('--model',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True);p.add_argument('--fastpath',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 runs=json.loads((CODE/'configs/training_runs.json').read_text());cfg=runs[a.run];out=a.output.resolve();data=a.data_root.resolve()
 if out.is_relative_to(data) or out.is_relative_to(CODE):p.error('Use an output directory outside the archives.')
 if out.exists():p.error('Refusing to reuse an existing training output directory.')
 prepared=(data/cfg['prepared']).resolve();assert prepared.is_file();records=json.loads(prepared.read_text())
 assert len(records)==(12 if cfg['task']=='T2' else 8)
 agent=CODE/'training/implementations'/cfg['agent'];plans=[]
 for rd,original in enumerate(cfg['commands'],1):
  dest=out/f'round_{rd:02}';values=dict(MODEL=str(a.model.resolve()),AGENT=str(agent),PREPARED=str(prepared),ADAPTER_OUTPUT=str(dest/'adapter'),CHECKPOINT_OUTPUT=str(dest/'checkpoints'))
  cmd=[sys.executable,'-u',str(CODE/'training/train_round.py')]+[values.get(x,x) for x in original]
  assert not any('/data/' in x for x in cmd if x not in values.values() and x not in cmd[:3])
  plans.append(dict(round=rd,command=cmd,previous_reference=str(out/f'round_{rd-1:02}/adapter') if rd>1 else None))
 if not a.execute:
  print(json.dumps(dict(run=a.run,base_model=cfg['model'],fixed_revision=cfg['revision'],prepared_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),rounds=plans,execution=False),indent=2));return
 for path in [a.model,a.overlay,a.fastpath]:
  if not path.is_dir():raise FileNotFoundError(path)
 if not (a.overlay/'openrlhf').is_dir():raise FileNotFoundError('Run training/prepare_runtime.py first.')
 expected=json.loads((CODE/'configs/base_model_fingerprints.json').read_text())[cfg['model']]
 if hashlib.sha256((a.model/'config.json').read_bytes()).hexdigest()!=expected['config_sha256']:raise ValueError('Base model config does not match the archived model')
 model_manifest=a.model/'model_manifest.json'
 if model_manifest.is_file():
  provenance=json.loads(model_manifest.read_text())
  if provenance.get('revision')!=expected['revision']:raise ValueError('Base model revision mismatch')

 out.mkdir(parents=True);(out/'invocations.json').write_text(json.dumps(plans,indent=2)+'\n')
 env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join(map(str,[a.overlay.resolve(),agent.parent,CODE/'training/implementations/main',a.fastpath.resolve()])),M2_RAY_CPUS='16',M2_RAY_GPUS='1',OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',M2_VLLM_KV_CACHE_BYTES='2147483648',M2_VLLM_MAX_NUM_SEQS='8',M2_VLLM_MAX_BATCHED_TOKENS='512',M2_FORMAL_ADMISSIBLE='1' if cfg['kind']=='main' else '0')
 env.update(TRITON_CACHE_DIR=str(out/'cache/triton'),FLASHINFER_WORKSPACE_BASE=str(out/'cache/flashinfer'))
 for plan in plans:
  rd=plan['round'];dest=out/f'round_{rd:02}';dest.mkdir();temp=dest/'tmp';temp.mkdir();env.update(TMPDIR=str(temp),M2_PILOT_OUTPUT=str(dest),M2_EDGE_OUTPUT=str(dest))
  for key in ['M2_ACTOR_ADAPTER','M2_REFERENCE_ADAPTER']:
   if plan['previous_reference']:env[key]=plan['previous_reference']
   else:env.pop(key,None)
  (dest/'invocation.json').write_text(json.dumps(plan,indent=2)+'\n')
  with (dest/'train.log').open('w') as log:subprocess.run(plan['command'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  text=re.sub(r'\x1b\[[0-9;]*m','',(dest/'train.log').read_text());steps=[int(x) for x in re.findall(r'Global step (\d+):',text)][-4:]
  if steps!=[1,2,3,4] or not (dest/'adapter/adapter_model.safetensors').is_file():raise RuntimeError(f'Round {rd}: missing optimizer steps or adapter')
  if 'M2_EFFECTIVE_WEIGHT_SYNC_OK' not in text:raise RuntimeError('Missing effective actor/vLLM weight synchronization marker')
  (dest/'release_step_check.json').write_text(json.dumps(dict(steps=steps,effective_weight_sync=True))+'\n')
  if cfg['kind']=='main':
   env['ENERGY_MODEL']=str(a.model.resolve())
   hold=[sys.executable,str(CODE/'evaluation/main/evaluate_reward_holdout.py'),'--formal','--output',str(dest/'heldout'),'--prepared',str(prepared.parent),'--task',cfg['task'],'--adapter',str(dest/'adapter'),'--samples-per-source','16','--batch-size','4','--seed',str(20262003+rd*1000)]
   subprocess.run(hold,env=env,check=True)
   if rd==1:
    metric=json.loads((dest/'heldout/metrics.json').read_text());rows=[ast.literal_eval(x) for x in re.findall(r'Global step \d+: (\{[^\n]*\})',text)][-4:]
    checks=dict(heldout_reward_finite=math.isfinite(metric["mean_reward"]),optimizer_steps_exact=steps==[1,2,3,4],invalid_mass_le_002=metric['invalid_rate']<=.02,kl_abs_max_lt_005=max(abs(x['kl']) for x in rows)<.05,gradient_max_lt_1=max(x['actor_grad_norm'] for x in rows)<1,directed_coverage_ge_020=metric['observed_directed_edges']/metric['possible_directed_edges']>=.2)
    (dest/'formal_round1_gate.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not all(checks.values()):raise RuntimeError('Formal round-one runtime gate failed')

 (out/'complete.json').write_text(json.dumps(dict(rounds=8,run=a.run,status='complete',base_model=cfg['model'],revision=cfg['revision']))+'\n')
if __name__=='__main__':main()
