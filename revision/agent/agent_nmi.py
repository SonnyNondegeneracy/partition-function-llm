"""OpenRLHF single-turn executor for the revision runs (same hook as main/m2_agent.py)."""
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from openrlhf.utils.agent import SingleTurnAgentExecutor
from nmi_protocol import score_response


class AgentExecutor(SingleTurnAgentExecutor):
    def __init__(self):
        super().__init__(None)

    async def execute(self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images=None):
        info = json.loads(label)
        assert hashlib.sha256(prompt.encode()).hexdigest() == info['rendered_prompt_sha256']
        assert sampling_params.temperature == 1. and sampling_params.top_p == 1.
        assert sampling_params.top_k in (-1, 0) and sampling_params.repetition_penalty == 1.
        assert sampling_params.max_tokens == info['max_tokens']
        sampling_params = deepcopy(sampling_params)
        sampling_params.logprobs = 1
        out = await super().execute(prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images)
        start, end = out['action_ranges'][0]
        ids = out['observation_tokens'][start:end]
        terminals = set(info.get('terminal_token_ids') or [hf_tokenizer.eos_token_id])
        text_ids = ids[:-1] if ids and ids[-1] in terminals else ids  # strip only one terminal token
        text = hf_tokenizer.decode(text_ids, skip_special_tokens=False)
        result = score_response(text, out['truncated'], info)
        out.update(reward=result['reward'], scores=result['reward'],
                   extra_logs={'valid_format': float(result['target'] is not None), 'semantic_reward': result['u']})
        record = {**info, **result, 'output': text, 'token_ids': ids, 'truncated': out['truncated'],
                  'rollout_log_probs': out['rollout_log_probs'], 'temperature': sampling_params.temperature}
        path = Path(os.environ['M2_PILOT_OUTPUT']) / f'rollouts_{os.getpid()}.jsonl'
        with path.open('a') as f:
            f.write(json.dumps(record) + '\n')
        return out
