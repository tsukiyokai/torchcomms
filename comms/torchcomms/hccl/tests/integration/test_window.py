# Pytest wrapper for HIXL one-sided (TorchCommWindow) cases.
# Mirrors test_collectives.py / test_aclgraph.py shape; isolated file
# because window cases need libcann_hixl runtime + a unique port range
# for the HIXL engine bootstrap (50000 + rank by default; collisions
# avoided by per-test pytest invocation).

import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).parent / "_2rank_runner.py"
WINDOW_CASES = [
    "window_put",   # put + signal + wait_signal end-to-end with numerics check
]

_PORT = 29800


@pytest.mark.parametrize("case", WINDOW_CASES)
def test_window_2rank(case):
    global _PORT
    _PORT += 1
    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_addr=127.0.0.1",
        f"--master_port={_PORT}",
        str(RUNNER),
        case,
    ]
    env = os.environ.copy()
    env.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
    env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True,
                         timeout=180)
    assert res.returncode == 0, (
        f"window {case} failed (rc={res.returncode}):\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
    assert res.stdout.count(f"{case} OK") == 2, \
        f"{case}: stdout missing rank acknowledgements:\n{res.stdout}"
