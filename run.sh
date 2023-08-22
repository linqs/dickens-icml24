#!/usr/bin/env bash

# Run all the experiments.

export PYTORCH_ENABLE_MPS_FALLBACK=1

function main() {
    trap exit SIGINT

#    pushd . >/dev/null
#      cd mnist_addition
#      python3 ./scripts/run_mnistadd1.py
#    popd >/dev/null

    pushd . >/dev/null
      cd mnist_addition
      python3 ./scripts/run_mnistadd2.py
    popd >/dev/null
}

main "$@"
