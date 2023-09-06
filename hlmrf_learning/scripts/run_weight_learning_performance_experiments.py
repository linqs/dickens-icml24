#!/usr/bin/env python3

import json
import os
import sys
import signal

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
RESULTS_BASE_DIR = os.path.join(THIS_DIR, "../results")
PSL_EXAMPLES_DIR = os.path.join(THIS_DIR, "../psl-examples")
PSL_EXTENDED_EXAMPLES_DIR = os.path.join(THIS_DIR, "../psl-extended-examples")
PERFORMANCE_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "performance")

STANDARD_EXPERIMENT_OPTIONS = {
    "runtime.log.level": "TRACE",
    "inference.normalize": "false",
    "weightlearning.inference": "DualBCDInference",
    "runtime.inference.method": "DualBCDInference",
    "gradientdescent.numsteps": "2500",
    "gradientdescent.scalestepsize": "false",
    "gradientdescent.batchgenerator": "FullBatchGenerator",
    "gradientdescent.movementbreak": "true",
    "gradientdescent.movementtolerance": "1.0e-3",
    "duallcqp.computeperiod": "10",
    "duallcqp.maxiterations": "10000"
}

STANDARD_DATASET_OPTIONS = {
    "epinions": {
        "duallcqp.primaldualthreshold": "0.1"
    },
    "citeseer": {
        "eval.closetruth": "true",
        "duallcqp.primaldualthreshold": "0.1"
    },
    "cora": {
        "eval.closetruth": "true",
        "duallcqp.primaldualthreshold": "0.1"
    },
    "yelp": {
        "duallcqp.primaldualthreshold": "1.0"
    },
}

DATASET_OPTION_RANGES = {
    "epinions": {
        "gradientdescent.stepsize": ["1.0e-3", "1.0e-2"],
        "duallcqp.regularizationparameter": ["1.0e-3"],
        "minimizer.objectivedifferencetolerance": ["0.1"]
    },
    "citeseer": {
        "gradientdescent.stepsize": ["1.0e-3", "1.0e-2"],
        "duallcqp.regularizationparameter": ["1.0e-3"],
        "minimizer.objectivedifferencetolerance": ["1.0"],
    },
    "cora": {
        "gradientdescent.stepsize": ["1.0e-3", "1.0e-2"],
        "duallcqp.regularizationparameter": ["1.0e-3"],
        "minimizer.objectivedifferencetolerance": ["1.0"]
    },
    "yelp": {
        "gradientdescent.stepsize": ["1.0e-3", "1.0e-2"],
        "duallcqp.regularizationparameter": ["1.0e-3"],
        "minimizer.objectivedifferencetolerance": ["1.0"]
    }
}

INFERENCE_OPTION_RANGES = {}

FIRST_ORDER_WL_METHODS = ["SquaredError", "BinaryCrossEntropy", "StructuredPerceptron", "Energy"]

FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES = {
    "gradientdescent.negativelogregularization": ["1.0e-3"],
    "gradientdescent.negativeentropyregularization": ["0.0"]
}

FIRST_ORDER_WL_METHODS_OPTION_RANGES = {
    "StructuredPerceptron": {
        "runtime.learn.method": ["StructuredPerceptron"]
    },
    "Energy": {
        "runtime.learn.method": ["Energy"]
    },
    "BinaryCrossEntropy": {
        "runtime.learn.method": ["BinaryCrossEntropy"],
        "minimizer.initialsquaredpenalty": ["2.0"],
        "minimizer.energylosscoefficient": ["0.0", "0.1", "1.0", "10.0"],
        "minimizer.proxvaluestepsize": ["1.0e-1", "1.0e-2"],
        "minimizer.squaredpenaltyincreaserate": ["2.0"],
        "minimizer.proxruleweight": ["1.0e-1", "1.0e-2", "1.0e-3"]
    },
    "SquaredError": {
        "runtime.learn.method": ["SquaredError"],
        "minimizer.initialsquaredpenalty": ["2.0"],
        "minimizer.energylosscoefficient": ["0.0", "0.1", "1.0", "10.0"],
        "minimizer.proxvaluestepsize": ["1.0e-1", "1.0e-2"],
        "minimizer.squaredpenaltyincreaserate": ["2.0"],
        "minimizer.proxruleweight": ["1.0e-1", "1.0e-2", "1.0e-3"]
    }
}


def enumerate_hyperparameters(hyperparameters_dict: dict, current_hyperparameters={}):
    for key in sorted(hyperparameters_dict):
        hyperparameters = []
        for value in hyperparameters_dict[key]:
            next_hyperparameters = current_hyperparameters.copy()
            next_hyperparameters[key] = value

            remaining_hyperparameters = hyperparameters_dict.copy()
            remaining_hyperparameters.pop(key)

            if remaining_hyperparameters:
                hyperparameters = hyperparameters + enumerate_hyperparameters(remaining_hyperparameters, next_hyperparameters)
            else:
                hyperparameters.append(next_hyperparameters)
        return hyperparameters


def run_first_order_wl_methods(dataset: str):
    base_out_dir = os.path.join(PERFORMANCE_RESULTS_DIR, dataset, "first_order_wl_methods")
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_cli_path = os.path.join(PSL_EXAMPLES_DIR, "{}/cli".format(dataset))
    dataset_json_path = os.path.join(dataset_cli_path, "{}.json".format(dataset))

    dataset_original_json = None
    with open(dataset_json_path, "r") as file:
        dataset_original_json = json.load(file)

    original_options = dataset_original_json["options"]

    # If the dataset is epinions, citeseer, or cora, then we need to load the extended json file.
    if dataset in ["epinions", "citeseer", "cora"]:
        dataset_extended_json_path = os.path.join(PSL_EXTENDED_EXAMPLES_DIR, "{}/cli/{}.json".format(dataset, dataset))

        with open(dataset_extended_json_path, "r") as file:
            dataset_json = json.load(file)
    else:
        dataset_json = dataset_original_json

    standard_experiment_option_ranges = {**INFERENCE_OPTION_RANGES,
                                         **FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES,
                                         **DATASET_OPTION_RANGES[dataset]}

    for method in FIRST_ORDER_WL_METHODS:
        for split in range(0, 5):
            method_out_dir = os.path.join(base_out_dir, "{}/split::{}".format(method, split))
            os.makedirs(method_out_dir, exist_ok=True)

            # Iterate over every combination options values.
            method_options_dict = {**standard_experiment_option_ranges,
                                   **FIRST_ORDER_WL_METHODS_OPTION_RANGES[method]}
            for options in enumerate_hyperparameters(method_options_dict):
                experiment_out_dir = method_out_dir
                for key, value in sorted(options.items()):
                    experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))
                os.makedirs(experiment_out_dir, exist_ok=True)

                if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                    print("Skipping experiment: {}.".format(experiment_out_dir))
                    continue

                dataset_json.update({"options":{**original_options,
                                                **STANDARD_EXPERIMENT_OPTIONS,
                                                **STANDARD_DATASET_OPTIONS[dataset],
                                                **options,
                                                "runtime.learn.output.model.path": "./{}_learned.psl".format(dataset)}})

                # Write the options the json file.
                with open(dataset_json_path, "w") as file:
                    json.dump(dataset_json, file, indent=4)

                # SED the split data.
                os.system("sed -i 's#/[0-9]/#/{}/#g' {}".format(split, dataset_json_path))

                # Run the experiment. Timeout after 2 hours.
                print("Running experiment: {}.".format(experiment_out_dir))

                def timeoutHandler(signum, frame):
                    print("Experiment Timeout: {}.".format(experiment_out_dir))

                    raise Exception("Timeout")

                signal.signal(signal.SIGALRM, timeoutHandler)
                signal.alarm(7200)

                try:
                    exit_code = os.system("cd {} && ./run.sh {} > out.txt 2> out.err".format(dataset_cli_path, experiment_out_dir))
                except Exception as e:
                    if e.args[0] != "Timeout":
                        raise e

                    exit_code = 0

                signal.alarm(0)

                # Save the output and json file. Some of these may fail if the experiment failed.
                os.system("mv {} {}".format(os.path.join(dataset_cli_path, "out.txt"), experiment_out_dir))
                os.system("mv {} {}".format(os.path.join(dataset_cli_path, "out.err"), experiment_out_dir))
                os.system("cp {} {}".format(os.path.join(dataset_cli_path, "{}.json".format(dataset)), experiment_out_dir))
                os.system("cp {} {}".format(os.path.join(dataset_cli_path, "{}_learned.psl".format(dataset)), experiment_out_dir))

                if exit_code != 0:
                    print("Experiment failed: {}.".format(experiment_out_dir))
                    exit()

                # Reset the json file.
                dataset_original_json.update({"options": original_options})
                with open(dataset_json_path, "w") as file:
                    json.dump(dataset_original_json, file)

                print("Finished experiment: Dataset:{}, Weight Learning Method: {}.".format(dataset, method))


def parse_args() -> str:
    if len(sys.argv) != 2:
        print("Usage: python3 run_weight_learning_performance_experiments.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]
    return dataset


def main():
    dataset = parse_args()

    run_first_order_wl_methods(dataset)


if __name__ == '__main__':
    main()
