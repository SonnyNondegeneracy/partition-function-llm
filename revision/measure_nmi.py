"""Sample one policy on all sources of a prepared T1/T2 task with vLLM and write counts + raw samples.

  python revision/measure_nmi.py --model BASE_DIR [--adapter ROUND_DIR/adapter] \
      --prepared PREPARED.json --samples 4096 --seed S --output OUT_DIR [--score-logprobs]

Outputs (OUT_DIR): counts.json (schema of analysis/common.load_counts: metrics.T1.{n_states,
edges,counts,invalid_counts,attempts}), samples.jsonl (source, sample, text, token_ids, target,
invalid_reason, reward, and with --score-logprobs: logp_policy, logp_base = summed log-probabilities
of the response tokens under the policy (base+adapter) and under the base model, both by teacher
forcing), run.json (arguments, versions, sha256 of inputs).
Sampling: temperature 1, top_p 1, per-source seed = seed + 10000*source. Prompts are tokenised with
add_special_tokens=False, exactly as in training (the rendered prompt already holds BOS if any).
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'agent'))
from nmi_protocol import score_response


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path, default=None)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--samples', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--score-logprobs', action='store_true')
    p.add_argument('--max-model-len', type=int, default=1024)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    p.add_argument('--llm-kwargs', default='{}', help='extra JSON kwargs for vllm.LLM (e.g. limit_mm_per_prompt)')
    a = p.parse_args()
    if a.output.exists():
        p.error('output exists')
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    import vllm, transformers
    recs = json.loads(a.prepared.read_text())
    labels = [json.loads(r['label']) for r in recs]
    llm = LLM(model=str(a.model), enable_lora=True, max_lora_rank=16, max_loras=1, seed=a.seed,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
              **{'dtype': 'bfloat16', 'language_model_only': True, **json.loads(a.llm_kwargs)})  # as evaluation/supplementary/measure.py
    tok = llm.get_tokenizer()
    lora = LoRARequest('policy', 1, str(a.adapter)) if a.adapter else None
    a.output.mkdir(parents=True)
    n = len(recs)
    counts = {}
    invalid = [0] * n
    attempts = [0] * n
    t0 = time.time()
    with (a.output / 'samples.jsonl').open('w') as f:
        for r, lab in zip(recs, labels):
            i = int(lab['source'])
            assert hashlib.sha256(r['input'].encode()).hexdigest() == lab['rendered_prompt_sha256']
            ids = tok(r['input'], add_special_tokens=False)['input_ids']
            sp = SamplingParams(n=a.samples, temperature=1.0, top_p=1.0, max_tokens=lab['max_tokens'],
                                seed=a.seed + 10000 * i)
            outs = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, lora_request=lora, use_tqdm=False)[0].outputs
            terminals = set(lab.get('terminal_token_ids') or [tok.eos_token_id])
            rows = []
            for k, o in enumerate(outs):
                tid = list(o.token_ids)
                text_ids = tid[:-1] if tid and tid[-1] in terminals else tid
                text = tok.decode(text_ids, skip_special_tokens=False)
                s = score_response(text, o.finish_reason == 'length', lab)
                attempts[i] += 1
                if s['target'] is None:
                    invalid[i] += 1
                else:
                    counts[(i, s['target'])] = counts.get((i, s['target']), 0) + 1
                rows.append(dict(source=i, sample=k, text=text, token_ids=tid, finish_reason=o.finish_reason, **s))
            if a.score_logprobs:  # teacher-forced log p(response) under policy and base
                seqs = [TokensPrompt(prompt_token_ids=ids + row['token_ids']) for row in rows]
                spl = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0)
                for key, req in (('logp_policy', lora), ('logp_base', None)):
                    if key == 'logp_policy' and lora is None:
                        continue
                    res = llm.generate(seqs, spl, lora_request=req, use_tqdm=False)
                    for row, q in zip(rows, res):
                        pl = q.prompt_logprobs
                        full = ids + row['token_ids']
                        row[key] = float(sum(pl[t][full[t]].logprob for t in range(len(ids), len(full))))
                if lora is None:
                    for row in rows:
                        row['logp_policy'] = row['logp_base']
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f'source {i}: valid {attempts[i]-invalid[i]}/{attempts[i]}  {time.time()-t0:.0f}s', flush=True)
    edges = sorted({(int(l['source']), int(v)) for l in labels for v in l['mapping'].values() if int(v) != int(l['source'])})
    metrics = dict(n_states=n, edges=[list(e) for e in edges], counts=[counts.get(e, 0) for e in edges],
                   invalid_counts=invalid, attempts=attempts)
    task = 'T2' if str(labels[0].get('task_id', 'T1')).startswith('T2') else 'T1'
    (a.output / 'counts.json').write_text(json.dumps({'metrics': {task: metrics}}, indent=1) + '\n')
    (a.output / 'run.json').write_text(json.dumps(dict(args={k: str(v) for k, v in vars(a).items()},
        vllm=vllm.__version__, transformers=transformers.__version__, prepared_sha256=sha(a.prepared),
        adapter_sha256=sha(a.adapter / 'adapter_model.safetensors') if a.adapter else None,
        seconds=time.time() - t0), indent=1) + '\n')
    print('MEASURE_NMI_DONE', a.output)


if __name__ == '__main__':
    main()
