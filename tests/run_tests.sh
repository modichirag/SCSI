#!/bin/sh
# Run the SCSI smoke tests. Activate your python env before invoking.
# stdout + stderr go to tests/logs/<test>.log; on failure the tail of the
# log is also echoed so the error is visible in the terminal.
#
# Usage (from any cwd):
#   ./tests/run_tests.sh
#   ./tests/run_tests.sh test_synthetic.py        # run a single test

set -u  # do NOT set -e: we want to keep going past a failing test

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
LOG_DIR="$REPO_ROOT/tests/logs"
mkdir -p "$LOG_DIR"

if [ "$#" -gt 0 ]; then
  TESTS="$@"
else
  TESTS="test_synthetic.py test_image.py"
fi

PASS=0
FAIL=0
for t in $TESTS; do
  log="$LOG_DIR/${t%.py}.log"
  echo "=== running $t -> $log ==="
  if python -u "$REPO_ROOT/tests/$t" >"$log" 2>&1; then
    echo "  PASS"
    PASS=$((PASS+1))
  else
    rc=$?
    echo "  FAIL (exit $rc). Last 100 lines:"
    tail -n 100 "$log" | sed 's/^/    /'
    FAIL=$((FAIL+1))
  fi
done

echo "---"
echo "summary: $PASS passed, $FAIL failed"
exit $FAIL
