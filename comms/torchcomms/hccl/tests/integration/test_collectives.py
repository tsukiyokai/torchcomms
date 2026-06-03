# Pytest wrapper that launches torchrun for each collective.
# Requires NPU >=2 cards to be visible.

import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).parent / "_2rank_runner.py"
COLLECTIVES = [
    "all_reduce",
    "broadcast",
    "all_gather_single",
    "reduce_scatter_single",
    "send_recv",
    "barrier",
    "gather",
    "all_to_all_v_single",
    # "split" — implemented but currently hangs on cann 9.0 build
    # (HcclCreateSubCommConfig is HCOMM_WEAK_SYMBOL; behaviour unstable);
    # _2rank_runner.py still has cmd_split for manual testing.
]

_PORT = 29600  # bumped per-test to avoid TCPStore reuse on consecutive runs


@pytest.mark.parametrize("op", COLLECTIVES)
def test_collective_2rank(op):
    global _PORT
    _PORT += 1
    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_addr=127.0.0.1",
        f"--master_port={_PORT}",
        str(RUNNER),
        op,
    ]
    env = os.environ.copy()
    env.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
    env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True,
                         timeout=120)
    assert res.returncode == 0, (
        f"collective {op} failed (rc={res.returncode}):\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
    )
    # Expect both ranks to print "[rank N] op OK".
    assert res.stdout.count(f"{op} OK") == 2, \
        f"{op}: stdout missing rank acknowledgements:\n{res.stdout}"
