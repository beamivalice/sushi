#!/bin/sh
# sushi-test: tier=tool
exec python3 "$(dirname "$0")/tape.py" "$@"
