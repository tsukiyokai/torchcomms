# Pytest wrapper that launches torchrun for each aclgraph case.
# Mirrors test_collectives.py shape; isolated file because aclgraph cases
# need NPUGraph + a longer timeout and are gated by HCCL+aclmdlRICapture
# compatibility (probe-verified on cann 9.0, see pyhccl/_probe_capture.py).

import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).parent / "_2rank_runner.py"
ACLGRAPH_CASES = [
    "aclgraph_single",   # 1 collective (all_reduce) captured + 3 replays
    "aclgraph_multi",    # 4 collectives in one graph + 3 replays
]

_PORT = 29700  # bumped per-test to avoid TCPStore reuse on consecutive runs


@pytest.mark.parametrize("case", ACLGRAPH_CASES)
def test_aclgraph_2rank(case):
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
        f"aclgraph case {case} failed (rc={res.returncode}):\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
    assert res.stdout.count(f"{case} OK") == 2, \
        f"{case}: stdout missing rank acknowledgements:\n{res.stdout}"
