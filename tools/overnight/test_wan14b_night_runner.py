import csv
import pytest
from wan14b_night_runner import check_rows


@pytest.mark.parametrize('parity,reward_std,grad,valid', [
    (0, 1, 0.01, True), (1e-12, 1, 0.01, False),
    (0, 0, 0.01, False), (0, 1, 0, False), (0, 1, float('nan'), False),
])
def test_night_gate_rejects_invalid_training(tmp_path, parity, reward_std, grad, valid):
    path = tmp_path/'metrics.csv'
    row = dict(loss=0.0, reward_mean=-1.0, reward_std=reward_std, grad_norm=grad,
               pre_update_logprob_abs_diff_max=parity)
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    if valid:
        assert len(check_rows(path)) == 1
        with pytest.raises(RuntimeError, match='Only 1'):
            check_rows(path, minimum=2)
    else:
        with pytest.raises(RuntimeError):
            check_rows(path)
