#!/bin/bash
# Optional setup: initialise a local git repo so the rollback parts of the
# exercise (Parts 3 and 4) can use `git checkout inventory-v1` as written.
#
# You do NOT need a GitHub account or any global git config. The script sets
# repo-local user.email and user.name so commits work in this folder.
#
# Skip this script if you only want to observe prompt-driven answer changes
# (Parts 1 and 2) without practising rollback.

set -e
cd "$(dirname "$0")"

if [ -d ".git" ]; then
    echo "Local git repo already initialised. Skipping."
    exit 0
fi

git init -q
git config user.email "learner@local"
git config user.name "learner"
git add .
git commit -q -m "Baseline (inventory-v1)"
git tag inventory-v1

echo "Initialised local git repo and tagged 'inventory-v1'."
echo "Run 'git tag' to confirm."
