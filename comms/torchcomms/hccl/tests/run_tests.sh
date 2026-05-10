#!/bin/bash
# Wrapper that runs the hccl backend pytest suites with the right
# `--import-mode=importlib` flag (otherwise pytest treats the source
# tree's comms/torchcomms/__init__.py as a package root and tries to
# import _comms.so from the source layout — which doesn't exist; the
# .so lives in the install layout).
#
# Usage from repo root:
#     ./comms/torchcomms/hccl/tests/run_tests.sh           # all
#     ./comms/torchcomms/hccl/tests/run_tests.sh unit      # unit only
#     ./comms/torchcomms/hccl/tests/run_tests.sh integration # integration only
#     ./comms/torchcomms/hccl/tests/run_tests.sh -k all_reduce  # filter

set -e

# Ensure environment is activated. If ASCEND_HOME_PATH is unset the user
# forgot to source set_env.sh; warn but continue (some unit tests may still pass).
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "warn: ASCEND_HOME_PATH not set; did you source the cann set_env.sh?" >&2
fi
if [ -z "${PYTHONPATH:-}" ] || [[ "$PYTHONPATH" != *"/install"* ]]; then
    echo "warn: PYTHONPATH does not point at <repo>/install" >&2
    echo "      did you cp build-hccl/comms/torchcomms/_comms_hccl.*.so to install/torchcomms/?" >&2
fi

case "${1:-all}" in
    unit)
        shift
        exec python3 -m pytest --import-mode=importlib \
            comms/torchcomms/hccl/tests/unit/ -v "$@"
        ;;
    integration)
        shift
        exec python3 -m pytest --import-mode=importlib \
            comms/torchcomms/hccl/tests/integration/ -v "$@"
        ;;
    all)
        exec python3 -m pytest --import-mode=importlib \
            comms/torchcomms/hccl/tests/ -v
        ;;
    *)
        # Anything else (e.g. -k filter) — pass directly through.
        exec python3 -m pytest --import-mode=importlib \
            comms/torchcomms/hccl/tests/ -v "$@"
        ;;
esac
