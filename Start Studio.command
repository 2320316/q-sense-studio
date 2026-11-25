#!/bin/bash
set -e
cd "$(dirname "$0")"
test -x .venv/bin/python || python3 -m venv .venv
if ! cmp -s requirements.txt .venv/qs-requirements.txt; then
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv/qs-requirements.txt
fi
exec .venv/bin/python start.py "$@"
