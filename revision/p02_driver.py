"""Resumable driver: re-measure archived adapters (R0) and train + measure new runs (E3-E8, demo).

  python revision/p02_driver.py --work $W ref  GROUP            # R0: archived adapters, all 8 rounds
  python revision/p02_driver.py --work $W train RUN [--rounds N] # train (run_nmi.py) then measure base + rounds
  python revision/p02_driver.py --work $W status                 # what is done
Every measurement goes to $W/meas/<name>/<policy>/ and is skipped if counts.json already exists.
Seeds: seed = 930000000 + 10000000*k + 1000000*round (k = fixed index of the trajectory below; base uses round 0);
measure_nmi adds 10000*source.
"""
import argparse, json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CODE = HERE.parent
REF = {  # group -> (prepared, model key)
    'main_9b/T1': ('T1_main.json', 'qwen9b'), 'main_9b/T2': ('T2_main.json', 'qwen9b'),
    **{f'kl_scan/lambda_{x}_reinforce_baseline_full_qwen35_v2{s}': ('T1_main.json', 'qwen9b')
       for x, s in [('0p2', ''), ('0p3', ''), ('0p5', ''), ('0p7', ''), ('1p5', '_1gpu'), ('3', ''), ('5', '')]},
}
RUNS = json.loads((HERE / 'runs_p02.json').read_text())
ORDER = list(REF) + list(RUNS)            # fixed index -> seed block
MODELS = {'qwen9b': 'models/qwen9b', 'qwen4b': 'models/qwen4b', 'gemma4': 'models/gemma4'}
GEMMA_KW = '{}'  # language_model_only=True is the sampler default; add kwargs here only if Gemma4 needs them


def model_key(run):
    m = RUNS[run]['model']
    return 'gemma4' if 'gemma' in m else ('qwen4b' if '4B' in m else 'qwen9b')


def measure(W, name, policy, k, rnd, model, prepared, adapter=None, cot=False):
    out = W / 'meas' / name / policy
    if (out / 'counts.json').is_file():
        print('skip', out); return
    if out.exists():
        raise SystemExit(f'{out} exists without counts.json: inspect, then remove it and rerun')
    cmd = [sys.executable, str(HERE / 'measure_nmi.py'), '--model', str(W / MODELS[model]), '--prepared', str(HERE / 'prepared' / prepared),
           '--samples', '1024' if cot else '4096', '--seed', str(930000000 + 10000000 * k + 1000000 * rnd), '--output', str(out)]
    if adapter:
        cmd += ['--adapter', str(adapter)]
    if model == 'gemma4':
        cmd += ['--llm-kwargs', GEMMA_KW]
    if cot:
        cmd += ['--score-logprobs', '--max-model-len', '1024']
    log = W / 'logs' / f'meas_{name.replace("/", "_")}_{policy}.log'
    with log.open('w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    print('done', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, required=True); ap.add_argument('cmd'); ap.add_argument('name', nargs='?')
    ap.add_argument('--rounds', type=int, default=None); a = ap.parse_args()
    W = a.work.resolve(); (W / 'logs').mkdir(parents=True, exist_ok=True)
    if a.cmd == 'status':
        for n in ORDER:
            d = W / 'meas' / n
            done = sorted(p.parent.name for p in d.glob('*/counts.json')) if d.exists() else []
            print(f'{n:60s} {" ".join(done)}')
        return
    k = ORDER.index(a.name)
    if a.cmd == 'ref':
        prepared, model = REF[a.name]
        from huggingface_hub import snapshot_download
        local = W / 'adapters'
        snapshot_download('Nondegeneracy/partition-function-llm', repo_type='dataset', local_dir=str(local),
                          allow_patterns=[f'adapters/{a.name}/round_0*/*'])
        measure(W, a.name, 'base', k, 0, model, prepared)
        for m in range(1, 9):
            measure(W, a.name, f'r{m:02d}', k, m, model, prepared, local / 'adapters' / a.name / f'round_{m:02d}')
        return
    if a.cmd == 'train':
        cfg = RUNS[a.name]; model = model_key(a.name); cot = 'cot' in cfg['prepared']
        prepared = cfg['prepared'].split('/')[-1]
        rounds = a.rounds or len(cfg['commands'])
        tdir = W / 'train' / a.name
        if not (tdir / 'complete.json').is_file():
            if tdir.exists():
                raise SystemExit(f'{tdir} incomplete: read its logs, report, remove and rerun (do not resume a half round)')
            cmd = [sys.executable, str(HERE / 'run_nmi.py'), '--execute', '--run', a.name, '--runs-file', 'runs_p02.json',
                   '--model', str(W / MODELS[model]), '--prepared', str(HERE / cfg['prepared']), '--output', str(tdir),
                   '--overlay', str(W / 'overlay'), '--fastpath', str(W / 'fastpath'), '--rounds', str(rounds)]
            with (W / 'logs' / f'train_{a.name}.log').open('w') as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        measure(W, a.name, 'base', k, 0, model, prepared, cot=cot)
        for m in range(1, rounds + 1):
            measure(W, a.name, f'r{m:02d}', k, m, model, prepared, tdir / f'round_{m:02d}/adapter', cot=cot)
        return
    raise SystemExit('unknown command')


if __name__ == '__main__':
    main()
