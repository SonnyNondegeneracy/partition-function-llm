"""Admission checks for newly collected formal measurements (never diagnostics)."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_request(contract_path, policy_id, prepared, adapter, model):
    contract = json.loads(Path(contract_path).read_text())
    assert contract['schema_version'] == 'm2.formal_measurement.v1'
    policy = contract['policies'][policy_id]
    assert str(adapter or '') == str(policy['adapter'] or '')
    assert model == contract['model']
    assert digest(Path(model, 'config.json')) == contract['model_config_sha256']
    if adapter:
        assert digest(Path(adapter, 'adapter_model.safetensors')) == policy['adapter_sha256']
        assert digest(Path(adapter, 'adapter_config.json')) == policy['adapter_config_sha256']
    for task, values in contract['tasks'].items():
        assert digest(Path(prepared, f'{task}_prepared.json')) == values['prepared_sha256']
        assert digest(Path(__file__).parent / 'tasks' / f'{task}_direct_formal_v1.json') == values['task_sha256']
    assert digest(Path(__file__).parent / 'm2_protocol.py') == contract['parser_sha256']
    return {'formal_admissible': True, 'data_role': 'formal_checkpoint_measurement',
            'scientific_acceptance': 'pending_analysis', 'policy_id': policy_id,
            'formal_measurement_manifest_sha256': digest(contract_path),
            'training_manifest_hash': contract['training_manifest_hash'],
            'measurement_timing': 'retrospective_after_training',
            'policy': policy}
