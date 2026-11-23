"""Replay archived parsers and verify counts/invalid counts without a model."""
from pathlib import Path
import argparse,json,hashlib,importlib.util,time,sys
import numpy as np
CODE=Path(__file__).resolve().parents[1]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
PARSERS={g:module('parser_'+g,CODE/'evaluation'/g/('m2_protocol.py' if g=='main' else 'protocol.py')) for g in ['main','supplementary','extensions']}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();data=a.data_root.resolve();t=time.monotonic();reports=[]
 jobs=[]
 index=json.loads((data/'metadata/measurement_index.json').read_text())
 for entry in index:
  group,task=entry['group'],entry['task'];raw=data/entry['raw_file']
  if group=='main':prepared=data/f'metadata/main/config/prepared/{task}_prepared.json';kind='main'
  elif group=='extensions':prepared=data/'metadata/extensions/contracts/T1_prepared.json';kind='extensions'
  else:
   mode='thinking' if group=='thinking_qwen9b' else 'direct'
   prepared=data/f'metadata/supplementary/prepared/{task}_{mode}_prepared.json';kind='supplementary'
  metric=json.loads((data/entry['metric_file']).read_text())
  if entry.get('metric_selector'):
   selector=entry['metric_selector'];metric=next(x for x in metric[selector['key']] if x['round']==selector['round'])
  jobs.append((entry,raw,metric,prepared,kind))
 for entry,raw,metric,prepared,kind in jobs:
  labels=[json.loads(x['label']) for x in json.loads(prepared.read_text())];n=len(labels);edges=[tuple(x) for x in metric['edges']];lookup={x:i for i,x in enumerate(edges)}
  count=np.zeros(len(edges),int);invalid=np.zeros(n,int);attempts=np.zeros(n,int);reward=0.;seeds=set();total=0;h=hashlib.sha256()
  with raw.open('rb') as stream:
   for line in stream:
    h.update(line)
    if not line.strip():continue
    row=json.loads(line);i=int(row['source']);scored=PARSERS[kind].score_response(row['text'],bool(row['truncated']),labels[i]);j=scored['target'];attempts[i]+=1;total+=1;reward+=scored['reward']
    if j is None:invalid[i]+=1
    else:count[lookup[(i,int(j))]]+=1
    if 'seed' in row:seeds.add(int(row['seed']))
  np.testing.assert_array_equal(count,metric['counts']);np.testing.assert_array_equal(invalid,metric['invalid_counts']);np.testing.assert_array_equal(attempts,metric['attempts'] if 'attempts' in metric else np.bincount(np.array(edges)[:,0],weights=np.array(metric['counts']),minlength=n)+np.array(metric['invalid_counts']))
  expected_hash=metric.get('raw_sha256',metric.get('samples_sha256'))
  if expected_hash:assert h.hexdigest()==expected_hash
  if 'mean_reward' in metric:np.testing.assert_allclose(reward/total,metric['mean_reward'],atol=1e-12,rtol=0)
  if 'mean_edge_reward' in metric:np.testing.assert_allclose(reward/total,metric['mean_edge_reward'],atol=1e-12,rtol=0)
  reports.append(dict(measurement_id=entry['measurement_id'],reuses_measurement_id=entry.get('reuses_measurement_id'),path=str(raw.relative_to(data)),samples=total,sha256=h.hexdigest(),counts_match=True,invalid_match=True,unique_recorded_seeds=len(seeds) if seeds else None))
  if len(reports)%20==0:print('PASS',len(reports),'/',len(jobs),flush=True)
 unique={row['sha256']:row['samples'] for row in reports}
 result=dict(status='PASS',measurement_records=len(reports),unique_raw_files=len(unique),sample_uses=sum(row['samples'] for row in reports),unique_responses=sum(unique.values()),seconds=time.monotonic()-t,measurements=reports)
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='measurements'}))
if __name__=='__main__':main()
