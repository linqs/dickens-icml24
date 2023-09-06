#!/usr/bin/env bash

# Run all the experiments.

export PYTORCH_ENABLE_MPS_FALLBACK=1

PSL_DATASETS='citeseer'

function main() {
    trap exit SIGINT

    ./hlmrf_learning/scripts/setup_psl_examples.sh
    for dataset in $PSL_DATASETS; do
        echo "Running psl inference timing experiments on dataset: ${dataset}."
        python3 ./hlmrf_learning/scripts/run_inference_timing_experiments.py ${dataset}
    done

#    pushd . >/dev/null
#      cd mnist_addition
#      python3 ./scripts/run_mnistadd1.py
#    popd >/dev/null

#    pushd . >/dev/null
#      cd mnist_addition
#      python3 ./scripts/run_mnistadd2.py
#    popd >/dev/null
}

main "$@"
