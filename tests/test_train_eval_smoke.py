"""End-to-end smoke test: generate data -> train -> checkpoint -> reload -> eval.

This exercises the exact path that was broken before this test suite existed
(missing optimizer/scheduler glue, and torch>=2.6's weights_only=True default
breaking checkpoint reload) - nothing here was previously verified by any
automated test, only by hand-running the CLI.
"""
from conftest import run_train_heron


def test_train_then_load_and_eval(tmp_path, synthetic_datadir):
    train_result = run_train_heron([
        f'--datadir={synthetic_datadir}',
        '--nobj-avg=20',
        '--prefix=smoke_test',
        f'--workdir={tmp_path}',
        '--num-epoch=1',
        '--batch-size=4',
        '--log-level=warning',
    ])
    assert train_result.returncode == 0, train_result.stderr
    assert (tmp_path / 'model' / 'smoke_test_best.pt').exists()
    assert (tmp_path / 'model' / 'smoke_test_checkpoint.pt').exists()

    eval_result = run_train_heron([
        f'--datadir={synthetic_datadir}',
        '--nobj-avg=20',
        '--prefix=smoke_test',
        f'--workdir={tmp_path}',
        '--task=eval',
        '--load',
        '--log-level=warning',
    ])
    assert eval_result.returncode == 0, eval_result.stderr
    assert 'Traceback' not in eval_result.stderr
    assert 'TEST metrics' in eval_result.stdout
    assert 'test_Sharpe' in eval_result.stdout
