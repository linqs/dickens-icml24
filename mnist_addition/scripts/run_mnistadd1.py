#!/usr/bin/env python3

import json
import os
import re

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
MNIST_CLI_DIR = os.path.join(THIS_DIR, "../cli")
RESULTS_BASE_DIR = os.path.join(THIS_DIR, "../results")
PERFORMANCE_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "performance")

# SPLITS = ["0", "1", "2", "3", "4"]
SPLITS = ["0"]
# TRAIN_SIZES = ["0040", "0060", "0080"]
TRAIN_SIZES = ["0080"]
# OVERLAPS = ["0.00", "0.50", "1.00"]
OVERLAPS = ["0.00"]

STANDARD_EXPERIMENT_OPTIONS = {
    "inference.normalize": "false",
    "runtime.log.level": "TRACE",
    "gradientdescent.scalestepsize": "false",
    "weightlearning.inference": "DualBCDInference",
    "runtime.inference.method": "DualBCDInference",
    "gradientdescent.numsteps": "10000",
    "gradientdescent.runfulliterations": "false",
    "duallcqp.computeperiod": "10",
    "duallcqp.maxiterations": "10000",
}

STANDARD_DATASET_OPTIONS = {
    "mnist-addition": {
        "duallcqp.primaldualthreshold": "0.001"
    }
}

INFERENCE_OPTION_RANGES = {
    "duallcqp.regularizationparameter": ["1.0e-3"]
}

# FIRST_ORDER_WL_METHODS = ["Energy", "BinaryCrossEntropy"]
FIRST_ORDER_WL_METHODS = ["BinaryCrossEntropy"]

FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES = {
    "gradientdescent.stepsize": ["1.0e-3"],
    "gradientdescent.negativelogregularization": ["1.0e-3"],
    "gradientdescent.negativeentropyregularization": ["0.0"]
}

FIRST_ORDER_WL_METHODS_OPTION_RANGES = {
    "Energy": {
        "runtime.learn.method": ["Energy"]
    },
    "BinaryCrossEntropy": {
        "runtime.learn.method": ["BinaryCrossEntropy"],
        "minimizer.initialsquaredpenalty": ["1.0"],
        "minimizer.objectivedifferencetolerance": ["0.01"],
        "minimizer.proxruleweight": ["1.0e-1", "1.0e-2"],
        "minimizer.numinternaliterations": ["500"]
    }
}

NEURAL_NETWORK_OPTIONS = {
    "dropout": ["0.0", "0.1"]
}

BEST_HYPERPARAMETERS = {
    "Energy": {
        "0040": {
            "0.00": {
                "runtime.learn.method": "Energy",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "Energy",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "Energy",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        },
        "0060": {
            "0.00": {
                "runtime.learn.method": "Energy",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "Energy",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "Energy",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        },
        "0080": {
            "0.00": {
                "runtime.learn.method": "Energy",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "Energy",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "Energy",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        }
    },
    "BinaryCrossEntropy": {
        "0040": {
            "0.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.initialsquaredpenalty": "1.0",
                "minimizer.objectivedifferencetolerance": "0.001",
                "minimizer.proxruleweight": "1.0e-2",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.initialsquaredpenalty": "1.0",
                "minimizer.objectivedifferencetolerance": "0.001",
                "minimizer.proxruleweight": "1.0e-2",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.initialsquaredpenalty": "1.0",
                "minimizer.objectivedifferencetolerance": "0.001",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        },
        "0060": {
            "0.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.initialsquaredpenalty": "1.0",
                "minimizer.objectivedifferencetolerance": "0.001",
                "minimizer.proxruleweight": "1.0e-2",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.initialsquaredpenalty": "1.0",
                "minimizer.objectivedifferencetolerance": "0.001",
                "minimizer.proxruleweight": "1.0e-2",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.initialsquaredpenalty": "1.0",
                "minimizer.objectivedifferencetolerance": "0.001",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        },
        "0080": {
            "0.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.initialsquaredpenalty": "0.1",
                "minimizer.objectivedifferencetolerance": "0.001",
                "minimizer.proxruleweight": "1.0e-2",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.initialsquaredpenalty": "1.0",
                "minimizer.objectivedifferencetolerance": "0.001",
                "minimizer.proxruleweight": "1.0e-2",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.initialsquaredpenalty": "1.0",
                "minimizer.objectivedifferencetolerance": "1.0e-5",
                "minimizer.proxruleweight": "1.0e-2",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-3",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        }
    }
}

BEST_NEURAL_NETWORK_HYPERPARAMETERS = {
    "Energy": {
        "0040": {
            "0.00": {
                "dropout": "0.1"
            },
            "0.50": {
                "dropout": "0.0"
            },
            "1.00": {
                "dropout": "0.1"
            }
        },
        "0060": {
            "0.00": {
                "dropout": "0.0"
            },
            "0.50": {
                "dropout": "0.0"
            },
            "1.00": {
                "dropout": "0.1"
            }
        },
        "0080": {
            "0.00": {
                "dropout": "0.0"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        }
    },
    "BinaryCrossEntropy": {
        "0040": {
            "0.00": {
                "dropout": "0.0"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        },
        "0060": {
            "0.00": {
                "dropout": "0.0"
            },
            "0.50": {
                "dropout": "0.0"
            },
            "1.00": {
                "dropout": "0.1"
            }
        },
        "0080": {
            "0.00": {
                "dropout": "0.0"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        }
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


def set_data_path(dataset_json, split, train_size, overlap):
    # sed dataset paths
    for predicate in dataset_json["predicates"]:
        if "options" in dataset_json["predicates"][predicate]:
            if "entity-data-map-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])

            if "save-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["save-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["save-path"])
                dataset_json["predicates"][predicate]["options"]["save-path"] = \
                    re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), dataset_json["predicates"][predicate]["options"]["save-path"])
                dataset_json["predicates"][predicate]["options"]["save-path"] = \
                    re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), dataset_json["predicates"][predicate]["options"]["save-path"])

        if "targets" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]

            if "infer" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]

        if "truth" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]
                dataset_json["predicates"][predicate]["truth"]["learn"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]
                dataset_json["predicates"][predicate]["truth"]["learn"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]

            if "infer" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["infer"]]
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["infer"]]
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["infer"]]

        if "observations" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]
                dataset_json["predicates"][predicate]["observations"]["learn"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]
                dataset_json["predicates"][predicate]["observations"]["learn"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]

            if "infer" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]


def run_first_order_wl_methods_hyperparamter_search():
    base_out_dir = os.path.join(PERFORMANCE_RESULTS_DIR, "mnist-add1", "first_order_wl_methods")
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(MNIST_CLI_DIR, "neupsl_models/experiment::mnist-add1.json")

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    standard_experiment_option_ranges = {**INFERENCE_OPTION_RANGES,
                                         **FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES}

    for method in FIRST_ORDER_WL_METHODS:
        method_out_dir = os.path.join(base_out_dir, method)
        os.makedirs(method_out_dir, exist_ok=True)

        for train_size in TRAIN_SIZES:
            for overlap in OVERLAPS:
                for split in ["0"]:
                    split_out_dir = os.path.join(method_out_dir, "split::{}/train-size::{}/overlap::{}".format(split, train_size, overlap))
                    os.makedirs(split_out_dir, exist_ok=True)

                    # Iterate over every combination options values.
                    method_options_dict = {**standard_experiment_option_ranges,
                                           **FIRST_ORDER_WL_METHODS_OPTION_RANGES[method]}
                    for options in enumerate_hyperparameters(method_options_dict):
                        for dropout in NEURAL_NETWORK_OPTIONS["dropout"]:
                            experiment_out_dir = split_out_dir
                            for key, value in sorted(options.items()):
                                experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))
                            experiment_out_dir = os.path.join(experiment_out_dir, "dropout::{}".format(dropout))

                            os.makedirs(experiment_out_dir, exist_ok=True)

                            if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                                print("Skipping experiment: {}.".format(experiment_out_dir))
                                continue

                            dataset_json.update({"options":{**original_options,
                                                            **STANDARD_EXPERIMENT_OPTIONS,
                                                            **STANDARD_DATASET_OPTIONS["mnist-addition"],
                                                            **options,
                                                            "runtime.learn.output.model.path": "./mnist-addition_learned.psl"}})

                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["learning-rate"] = options["gradientdescent.stepsize"]
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["dropout"] = dropout

                            # Set the data path.
                            set_data_path(dataset_json, split, train_size, overlap)

                            # Write the options the json file.
                            with open(os.path.join(MNIST_CLI_DIR, "mnist-addition.json"), "w") as file:
                                json.dump(dataset_json, file, indent=4)

                            # Run the experiment.
                            print("Running experiment: {}.".format(experiment_out_dir))
                            exit_code = os.system("cd {} && ./run.sh {} > out.txt 2> out.err".format(MNIST_CLI_DIR, experiment_out_dir))

                            if exit_code != 0:
                                print("Experiment failed: {}.".format(experiment_out_dir))
                                exit()

                            # Save the output and json file.
                            os.system("mv {} {}".format(os.path.join(MNIST_CLI_DIR, "out.txt"), experiment_out_dir))
                            os.system("mv {} {}".format(os.path.join(MNIST_CLI_DIR, "out.err"), experiment_out_dir))
                            os.system("cp {} {}".format(os.path.join(MNIST_CLI_DIR, "mnist-addition.json"), experiment_out_dir))
                            os.system("cp {} {}".format(os.path.join(MNIST_CLI_DIR, "mnist-addition_learned.psl"), experiment_out_dir))
                            os.system("cp -r {} {}".format(os.path.join(MNIST_CLI_DIR, "inferred-predicates"), experiment_out_dir))


def run_first_order_wl_methods():
    base_out_dir = os.path.join(PERFORMANCE_RESULTS_DIR, "mnist-add1", "first_order_wl_methods")
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(MNIST_CLI_DIR, "neupsl_models/experiment::mnist-add1.json")

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    for method in FIRST_ORDER_WL_METHODS:
        method_out_dir = os.path.join(base_out_dir, method)
        os.makedirs(method_out_dir, exist_ok=True)

        for train_size in TRAIN_SIZES:
            for overlap in OVERLAPS:
                options = BEST_HYPERPARAMETERS[method][train_size][overlap]
                dropout = BEST_NEURAL_NETWORK_HYPERPARAMETERS[method][train_size][overlap]["dropout"]

                for split in SPLITS:
                    split_out_dir = os.path.join(method_out_dir, "split::{}/train-size::{}/overlap::{}".format(split, train_size, overlap))
                    os.makedirs(split_out_dir, exist_ok=True)

                    experiment_out_dir = split_out_dir
                    for key, value in sorted(options.items()):
                        experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))
                    experiment_out_dir = os.path.join(experiment_out_dir, "dropout::{}".format(dropout))

                    os.makedirs(experiment_out_dir, exist_ok=True)

                    if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                        print("Skipping experiment: {}.".format(experiment_out_dir))
                        continue

                    dataset_json.update({"options":{**original_options,
                                                    **STANDARD_EXPERIMENT_OPTIONS,
                                                    **STANDARD_DATASET_OPTIONS["mnist-addition"],
                                                    **options,
                                                    "runtime.learn.output.model.path": "./mnist-addition_learned.psl"}})

                    dataset_json["predicates"]["NeuralClassifier/2"]["options"]["learning-rate"] = options["gradientdescent.stepsize"]
                    dataset_json["predicates"]["NeuralClassifier/2"]["options"]["dropout"] = dropout

                    # Set the data path.
                    set_data_path(dataset_json, split, train_size, overlap)

                    # Write the options the json file.
                    with open(os.path.join(MNIST_CLI_DIR, "mnist-addition.json"), "w") as file:
                        json.dump(dataset_json, file, indent=4)

                    # Run the experiment.
                    print("Running experiment: {}.".format(experiment_out_dir))
                    exit_code = os.system("cd {} && ./run.sh {} > out.txt 2> out.err".format(MNIST_CLI_DIR, experiment_out_dir))

                    if exit_code != 0:
                        print("Experiment failed: {}.".format(experiment_out_dir))
                        exit()

                    # Save the output and json file.
                    os.system("mv {} {}".format(os.path.join(MNIST_CLI_DIR, "out.txt"), experiment_out_dir))
                    os.system("mv {} {}".format(os.path.join(MNIST_CLI_DIR, "out.err"), experiment_out_dir))
                    os.system("cp {} {}".format(os.path.join(MNIST_CLI_DIR, "mnist-addition.json"), experiment_out_dir))
                    os.system("cp {} {}".format(os.path.join(MNIST_CLI_DIR, "mnist-addition_learned.psl"), experiment_out_dir))
                    os.system("cp -r {} {}".format(os.path.join(MNIST_CLI_DIR, "inferred-predicates"), experiment_out_dir))

def main():
    # run_first_order_wl_methods_hyperparamter_search()
    run_first_order_wl_methods()


if __name__ == '__main__':
    main()
