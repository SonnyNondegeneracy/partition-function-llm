"""Build a prepared-prompt file for a new model family and/or a reasoning (CoT) answer style.

The user message is taken verbatim from an archived Qwen-rendered prepared file and re-rendered
with the target tokenizer's chat template (thinking disabled). Only the final instruction
sentence changes for --style cot. Labels keep mapping and rewards; rendered_prompt_sha256,
max_tokens, terminal_token_ids, parser style and model id are updated.

  python revision/prepare_variant.py --source PREPARED.json --tokenizer MODEL_DIR \
      --style direct|cot --max-tokens 32|512 --output OUT.json [--model-id ID]
Self-check: with a Qwen3.5 tokenizer and --style direct the output inputs must equal the source.
"""
import argparse, hashlib, json
from pathlib import Path

DIRECT = "Reply only ANSWER: <ID>."
COT = ("Think briefly about which candidate to choose (at most 80 words), "
       "then end with a final line of the form ANSWER: <ID>.")


def user_content(rendered):
    head, tail = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant"
    assert rendered.startswith(head) and tail in rendered, "source must be a Qwen-rendered prepared file"
    return rendered[len(head):rendered.index(tail)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--style", choices=["direct", "cot"], required=True)
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", default=None)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.tokenizer))
    gen = {}
    gpath = a.tokenizer / "generation_config.json"
    if gpath.is_file():
        gen = json.loads(gpath.read_text())
    eos = gen.get("eos_token_id", tok.eos_token_id)
    terminals = sorted({int(x) for x in (eos if isinstance(eos, list) else [eos]) if x is not None}
                       | ({int(tok.eos_token_id)} if tok.eos_token_id is not None else set()))
    src = json.loads(a.source.read_text())
    out, same = [], True
    for rec in src:
        content = user_content(rec["input"])
        assert content.count(DIRECT) == 1 and content.rstrip().endswith(DIRECT), "unexpected instruction sentence"
        if a.style == "cot":
            content = content.replace(DIRECT, COT)
        rendered = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        same &= rendered == rec["input"]
        label = json.loads(rec["label"])
        label.update(rendered_prompt_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
                     max_tokens=a.max_tokens, terminal_token_ids=terminals, parser=a.style,
                     reward_kind="edge" if "edge_rewards" in label else "state",
                     model_id=a.model_id or str(a.tokenizer), source_prepared=str(a.source.name))
        out.append({"input": rendered, "label": json.dumps(label, sort_keys=True)})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    n_tok = [len(tok(r["input"], add_special_tokens=False)["input_ids"]) for r in out]
    print(json.dumps(dict(output=str(a.output), records=len(out), style=a.style, terminals=terminals,
                          prompt_tokens_max=max(n_tok), identical_to_source=same,
                          sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))


if __name__ == "__main__":
    main()
