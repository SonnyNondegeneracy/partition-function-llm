"""Run one revision trajectory (8 cumulative LoRA rounds). Dry run unless --execute.

Same round semantics as training/run.py: each round loads the previous cumulative adapter as actor
and frozen reference, runs four optimizer steps and saves a new cumulative adapter.
Differences: runs come from revision/runs_p02.json, the prepared file is given explicitly,
the base-model fingerprint is recorded (not checked against the Qwen table), no held-out gate.
"""
from pathlib import Path
import argparse, json, os, sys, subprocess, hashlib, re
HERE = Path(__file__).resolve().parent
CODE = HERE.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ['--model', '--prepared', '--output', '--overlay', '--fastpath']:
        p.add_argument(k, type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--runs-file', default='runs_p02.json')
    p.add_argument('--rounds', type=int, default=None, help='default: all rounds of the run; smoke test: 1')
    a = p.parse_args()
    cfg = json.loads((HERE / a.runs_file).read_text())[a.run]
    out = a.output.resolve()
    if out.exists():
        p.error('Refusing to reuse an existing output directory.')
    prepared = a.prepared.resolve()
    records = json.loads(prepared.read_text())
    assert len(records) in (8, 12)
    agent = (CODE / cfg['agent']) if cfg['agent'].startswith('training/') else (HERE / cfg['agent'])
    n_steps = int(cfg['commands'][0][cfg['commands'][0].index('--train.num_episodes') + 1])
    plans = []
    for rd, original in enumerate(cfg['commands'][:a.rounds or len(cfg['commands'])], 1):
        dest = out / f'round_{rd:02}'
        values = dict(MODEL=str(a.model.resolve()), AGENT=str(agent), PREPARED=str(prepared),
                      ADAPTER_OUTPUT=str(dest / 'adapter'), CHECKPOINT_OUTPUT=str(dest / 'checkpoints'))
        cmd = [sys.executable, '-u', str(CODE / 'training/train_round.py')] + [values.get(x, x) for x in original]
        plans.append(dict(round=rd, command=cmd, previous_reference=str(out / f'round_{rd-1:02}/adapter') if rd > 1 else None))
    fp = dict(run=a.run, model=cfg['model'], config_sha256=hashlib.sha256((a.model / 'config.json').read_bytes()).hexdigest(),
              model_manifest=json.loads((a.model / 'model_manifest.json').read_text()) if (a.model / 'model_manifest.json').is_file() else None,
              prepared_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest())
    if not a.execute:
        print(json.dumps(dict(fingerprint=fp, rounds=plans, execution=False), indent=2))
        return
    if not (a.overlay / 'openrlhf').is_dir():
        raise FileNotFoundError('Run training/prepare_runtime.py first.')
    out.mkdir(parents=True)
    (out / 'fingerprint.json').write_text(json.dumps(fp, indent=2) + '\n')
    (out / 'invocations.json').write_text(json.dumps(plans, indent=2) + '\n')
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join(map(str, [a.overlay.resolve(), agent.parent, CODE / 'training/implementations/main', a.fastpath.resolve()])),
               M2_RAY_CPUS='16', M2_RAY_GPUS='1', OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
               M2_VLLM_KV_CACHE_BYTES='2147483648', M2_VLLM_MAX_NUM_SEQS='8', M2_VLLM_MAX_BATCHED_TOKENS='512', M2_FORMAL_ADMISSIBLE='0')
    env.update(cfg.get('env', {}))
    env.update(TRITON_CACHE_DIR=str(out / 'cache/triton'), FLASHINFER_WORKSPACE_BASE=str(out / 'cache/flashinfer'))
    for plan in plans:
        rd = plan['round']; dest = out / f'round_{rd:02}'; dest.mkdir(); temp = dest / 'tmp'; temp.mkdir()
        env.update(TMPDIR=str(temp), M2_PILOT_OUTPUT=str(dest), M2_EDGE_OUTPUT=str(dest))
        for key in ['M2_ACTOR_ADAPTER', 'M2_REFERENCE_ADAPTER']:
            if plan['previous_reference']:
                env[key] = plan['previous_reference']
            else:
                env.pop(key, None)
        (dest / 'invocation.json').write_text(json.dumps(plan, indent=2) + '\n')
        with (dest / 'train.log').open('w') as log:
            subprocess.run(plan['command'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        text = re.sub(r'\x1b\[[0-9;]*m', '', (dest / 'train.log').read_text())
        steps = [int(x) for x in re.findall(r'Global step (\d+):', text)][-n_steps:]
        if steps != list(range(1, n_steps + 1)) or not (dest / 'adapter/adapter_model.safetensors').is_file():
            raise RuntimeError(f'Round {rd}: missing optimizer steps or adapter')
        if 'M2_EFFECTIVE_WEIGHT_SYNC_OK' not in text:
            raise RuntimeError('Missing effective actor/vLLM weight synchronization marker')
        (dest / 'release_step_check.json').write_text(json.dumps(dict(steps=steps, effective_weight_sync=True)) + '\n')
    (out / 'complete.json').write_text(json.dumps(dict(rounds=len(plans), run=a.run, status='complete')) + '\n')


if __name__ == '__main__':
    main()
