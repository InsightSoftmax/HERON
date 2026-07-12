"""Regression test for the YAML config loader in train_heron.py.

The YAML files in config/ use hyphenated keys (lr-init, num-bilinear, ...)
but argparse dests use underscores (lr_init). parser.set_defaults(**yaml_args)
silently accepts hyphenated keys as dead dict entries instead of erroring,
so every value in a YAML config was being ignored with no error at all.
"""
import yaml

from conftest import run_train_heron


def test_yaml_config_values_actually_override_defaults(tmp_path, synthetic_datadir):
    config = {
        'lr-init': 0.0123456,  # distinctive value, not the coded default (0.001)
        'num-epoch': 1,
        'batch-size': 4,
    }
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))

    result = run_train_heron([
        f'--yaml={config_path}',
        f'--datadir={synthetic_datadir}',
        '--nobj-avg=20',
        '--prefix=yaml_override_test',
        f'--workdir={tmp_path}',
        '--log-level=info',
    ])

    assert result.returncode == 0, result.stderr
    assert 'lr_init: 0.0123456' in result.stdout
