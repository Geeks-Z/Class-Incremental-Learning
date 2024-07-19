import copy
import json
from loguru import logger
import os
import pickle
import random
import statistics
import sys
import time
from easydict import EasyDict as edict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import yaml
from inclearn.models import IncrementalLearner
from inclearn.lib import factory
from inclearn.lib import logger as logger_lib
from inclearn.lib import metrics, results_utils, utils


def train(args):
    logger_lib.set_logging_level(args["logging"])

    autolabel = _set_up_options(args)
    print(json.dumps(args, indent=4))
    if args["autolabel"]:
        args["label"] = autolabel

    if args["label"]:
        logger.info("Label: {}".format(args["label"]))
        try:
            os.system("echo '\ek{}\e\\'".format(args["label"]))
        except:
            pass
    if args["resume"] and not os.path.exists(args["resume"]):
        raise IOError(f"Saved model {args['resume']} doesn't exist.")

    if args["save_model"] != "never" and args["label"] is None:
        raise ValueError(
            f"Saving model every {args['save_model']} but no label was specified."
        )

    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    start_date = utils.get_date()

    orders = copy.deepcopy(args["order"])
    del args["order"]
    if orders is not None:
        assert isinstance(orders, list) and len(orders)
        assert all(isinstance(o, list) for o in orders)
        assert all([isinstance(c, int) for o in orders for c in o])
    else:
        orders = [None for _ in range(len(seed_list))]

    avg_inc_accs, last_accs, forgettings = [], [], []
    for i, seed in enumerate(seed_list):
        logger.warning("Launching run {}/{}".format(i + 1, len(seed_list)))
        args["seed"] = seed
        args["device"] = device

        start_time = time.time()

        for avg_inc_acc, last_acc, forgetting in _train(args, start_date, orders[i], i):
            yield avg_inc_acc, last_acc, forgetting, False

        avg_inc_accs.append(avg_inc_acc)
        last_accs.append(last_acc)
        forgettings.append(forgetting)

        logger.info("Training finished in {}s.".format(int(time.time() - start_time)))
        yield avg_inc_acc, last_acc, forgetting, True

    logger.info("Label was: {}".format(args["label"]))

    logger.info(
        "Results done on {} seeds: avg: {}, last: {}, forgetting: {}".format(
            len(seed_list),
            _aggregate_results(avg_inc_accs),
            _aggregate_results(last_accs),
            _aggregate_results(forgettings),
        )
    )
    logger.info(
        "Individual results avg: {}".format(
            [round(100 * acc, 2) for acc in avg_inc_accs]
        )
    )
    logger.info(
        "Individual results last: {}".format([round(100 * acc, 2) for acc in last_accs])
    )
    logger.info(
        "Individual results forget: {}".format(
            [round(100 * acc, 2) for acc in forgettings]
        )
    )

    logger.info(f"Command was {' '.join(sys.argv)}")


def _train(args, start_date, class_order, run_id):
    _set_global_parameters(args)
    results, results_folder = _set_results(args, start_date)
    args["results_folder"] = results_folder
    args["run_id"] = run_id
    inc_dataset, model = _set_data_model(args, class_order)

    memory, memory_val = None, None
    metric_logger = metrics.MetricLogger(
        inc_dataset.n_tasks, inc_dataset.n_classes, inc_dataset.increments
    )

    # logger
    dataset_name = args["dataset"]
    initial_increment = args["initial_increment"]
    increment = args["increment"]

    exp_name = args["label"]
    if "fast_dev_run" in args and args["fast_dev_run"]:
        exp_name += "-fast_dev_run"

    logger_cfg = dict(
        project=f"{dataset_name}-b{initial_increment}_{increment}",
        name=exp_name,
        notes=args.get("exp_notes", None),
        tags=args.get("tags", None),
    )

    wandb.init(config=args, **logger_cfg)
    wandb.run.log_code(".")
    # define our custom x axis metric
    wandb.define_metric("epoch")
    wandb.define_metric("task")
    # set all other train/ metrics to use this step
    wandb.define_metric("task_*", step_metric="epoch")
    wandb.define_metric("incremental_acc", step_metric="task")
    wandb.define_metric("average_acc", step_metric="task")
    wandb.define_metric("last_acc", step_metric="task")
    wandb.define_metric("incremental_acc_top5", step_metric="task")
    wandb.define_metric("last_acc_top5", step_metric="task")
    wandb.define_metric("forgetting", step_metric="task")

    for task_id in range(inc_dataset.n_tasks):
        if args["dataset"] != "core50":
            task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(
                memory, memory_val
            )
        else:
            # currently not supported memory
            task_info, train_loader, val_loader, test_loader = preparing_core50(
                inc_dataset, task_id, args
            )

        if task_info["task"] == args["max_task"]:
            break

        model.set_task_info(task_info)

        # ---------------
        # 1. Prepare Task
        # ---------------
        model.eval()
        model.before_task(train_loader, val_loader if val_loader else test_loader)

        # -------------
        # 2. Train Task
        # -------------
        n_params = sum(p.numel() for p in model.network.parameters())
        n_trainable_params = sum(
            p.numel() for p in model.network.parameters() if p.requires_grad
        )
        logger.info(f"Number of parameters: {n_params:,}")
        logger.info(f"Number of trainable parameters: {n_trainable_params:,}")
        train_task = _train_task(
            args,
            model,
            train_loader,
            val_loader,
            test_loader,
            run_id,
            task_id,
            task_info,
        )

        # ----------------
        # 3. Conclude Task
        # ----------------
        model.eval()
        _after_task(
            args, model, inc_dataset, run_id, task_id, results_folder, train_task
        )

        # ------------
        # 4. Eval Task
        # ------------
        logger.info("Eval on {}->{}.".format(0, task_info["max_class"]))
        ypreds, ytrue = model.eval_task(test_loader)
        task_size = task_info["increment"]
        if args["dataset"] == "core50":
            task_size = args["initial_increment"]
        metric_logger.log_task(
            ypreds, ytrue, task_size=task_size, zeroshot=args.get("all_test_classes")
        )

        if args["label"]:
            logger.info(args["label"])
        logger.info(
            "Avg inc acc: {}.".format(
                metric_logger.last_results["incremental_accuracy"]
            )
        )
        logger.info(
            "Avg acc: {}.".format(metric_logger.last_results["average_accuracy"])
        )
        logger.info("Current acc: {}.".format(metric_logger.last_results["accuracy"]))
        logger.info(
            "Avg inc acc top5: {}.".format(
                metric_logger.last_results["incremental_accuracy_top5"]
            )
        )
        logger.info(
            "Current acc top5: {}.".format(metric_logger.last_results["accuracy_top5"])
        )
        logger.info("Forgetting: {}.".format(metric_logger.last_results["forgetting"]))
        logger.info("Cord metric: {:.2f}.".format(metric_logger.last_results["cord"]))
        if task_id > 0:
            logger.info(
                "Old accuracy: {:.2f}, mean: {:.2f}.".format(
                    metric_logger.last_results["old_accuracy"],
                    metric_logger.last_results["avg_old_accuracy"],
                )
            )
            logger.info(
                "New accuracy: {:.2f}, mean: {:.2f}.".format(
                    metric_logger.last_results["new_accuracy"],
                    metric_logger.last_results["avg_new_accuracy"],
                )
            )
        if args.get("all_test_classes"):
            logger.info(
                "Seen classes: {:.2f}.".format(
                    metric_logger.last_results["seen_classes_accuracy"]
                )
            )
            logger.info(
                "unSeen classes: {:.2f}.".format(
                    metric_logger.last_results["unseen_classes_accuracy"]
                )
            )

        results["results"].append(metric_logger.last_results)
        _log_metrics(metric_logger.last_results, task_id)

        avg_inc_acc = results["results"][-1]["incremental_accuracy"]
        last_acc = results["results"][-1]["accuracy"]["total"]
        forgetting = results["results"][-1]["forgetting"]
        yield avg_inc_acc, last_acc, forgetting

        memory = model.get_memory()
        memory_val = model.get_val_memory()

        if args["label"] is not None:
            results_utils.save_results(
                results,
                args["label"],
                args["model"],
                start_date,
                run_id,
                args["seed"],
                args["dataset"],
                args["initial_increment"],
                args["increment"],
            )

    logger.info(
        "Average Incremental Accuracy: {}.".format(
            results["results"][-1]["incremental_accuracy"]
        )
    )

    del model
    del inc_dataset


# ------------------------
# Lifelong Learning phases
# ------------------------


def _train_task(
    config,
    model: IncrementalLearner,
    train_loader,
    val_loader,
    test_loader,
    run_id,
    task_id,
    task_info,
):
    train_task = True
    if (
        config["resume"] is not None
        and os.path.isdir(config["resume"])
        and ((config["resume_first"] and task_id == 0) or not config["resume_first"])
    ):
        load_success = model.load_parameters(config["resume"], run_id)
        if load_success:
            logger.info(
                "Skipping training phase {} because reloading pretrained model.".format(
                    task_id
                )
            )
        train_task = not load_success
    elif (
        config["resume"] is not None
        and os.path.isfile(config["resume"])
        and os.path.exists(config["resume"])
        and task_id == 0
    ):
        # In case we resume from a single model file, it's assumed to be from the first task.
        model.network = config["resume"]
        logger.info(
            "Skipping initial training phase {} because reloading pretrained model.".format(
                task_id
            )
        )
        train_task = False

    if train_task:
        logger.info(
            "Train on {}->{}.".format(task_info["min_class"], task_info["max_class"])
        )
        model.train()
        model.train_task(train_loader, val_loader if val_loader else test_loader)

    return train_task


def _after_task(
    config,
    model: IncrementalLearner,
    inc_dataset,
    run_id,
    task_id,
    results_folder,
    train_task,
):
    if (
        config["resume"]
        and (os.path.isdir(config["resume"]) or os.path.isfile(config["resume"]))
        and not config["recompute_meta"]
        and ((config["resume_first"] and task_id == 0) or not config["resume_first"])
        and not train_task
    ):
        model.load_metadata(config["resume"], run_id)
    else:
        model.after_task_intensive(inc_dataset)

    model.after_task(inc_dataset)

    if config["label"] and (
        config["save_model"] == "task"
        or (config["save_model"] == "last" and task_id == inc_dataset.n_tasks - 1)
        or (config["save_model"] == "first" and task_id == 0)
    ):
        model.save_parameters(results_folder, run_id)
        model.save_metadata(results_folder, run_id)


# ----------
# Parameters
# ----------


def _set_results(config, start_date):
    if config["label"]:
        results_folder = results_utils.get_save_folder(
            config["model"],
            start_date,
            config["label"],
            config["dataset"],
            config["initial_increment"],
            config["increment"],
        )
    else:
        results_folder = None

    if config["save_model"]:
        logger.info(
            "Model will be save at this rythm: {}.".format(config["save_model"])
        )

    results = results_utils.get_template_results(config)

    return results, results_folder


def preparing_core50(inc_dataset, task_id: int, args: dict):
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from inclearn.lib.data.incdataset import DummyDataset

    args = edict(args)
    train_dataset = inc_dataset.train[task_id]
    test_dataset = inc_dataset.test[0]  # only have single task for test set
    num_train = len(train_dataset)
    num_test = len(test_dataset)

    memory_flags_train = np.zeros(num_train)
    memory_flags_test = np.zeros(num_test)

    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    train_transforms = [*train_trsf, *common_trsf]
    test_transforms = [*test_trsf, *common_trsf]

    train_trsf = transforms.Compose(train_transforms)
    test_trsf = transforms.Compose(test_transforms)

    sampler = None
    batch_size = args.batch_size
    test_batch_size = args.get("test_batch_size", batch_size)
    num_workers = args.workers

    train_loader = DataLoader(
        DummyDataset(
            train_dataset._x,
            train_dataset._y,
            memory_flags_train,
            train_trsf,
            open_image=True,
        ),
        batch_size=batch_size,
        shuffle=True if sampler is None else False,
        num_workers=num_workers,
        batch_sampler=sampler,
    )
    test_loader = DataLoader(
        DummyDataset(
            test_dataset._x,
            test_dataset._y,
            memory_flags_test,
            test_trsf,
            open_image=True,
        ),
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        batch_sampler=sampler,
    )
    val_loader = None

    task_info = {
        "min_class": min(train_dataset._y),
        "max_class": max(train_dataset._y),
        "total_n_classes": len(np.unique(train_dataset._y)),
        "increment": 0,  # self.increments[self._current_task],
        "task": task_id,
        "max_task": 8,
        "n_train_data": len(train_dataset._y),
        "n_test_data": len(test_dataset._y),
    }

    return task_info, train_loader, val_loader, test_loader


def _set_data_model(config, class_order):
    if config["dataset"] != "core50":
        inc_dataset = factory.get_data(config, class_order)
        config["classes_order"] = inc_dataset.class_order
    else:
        from continuum.datasets import Core50
        from continuum.scenarios import ContinualScenario

        logger.info("Train on CORe50 with Domain Incremental Learning.")
        dataset_path = config["data_path"]
        train_dataset = Core50(
            dataset_path, scenario="domains", classification="object", train=True
        )
        test_dataset = Core50(
            dataset_path, scenario="domains", classification="object", train=False
        )

        train_scenario = ContinualScenario(train_dataset)
        test_scenario = ContinualScenario(test_dataset)

        n_tasks = len(train_scenario)
        increments = 0
        n_classes = 50

        inc_dataset = edict(
            train=train_scenario,
            test=test_scenario,
            n_tasks=n_tasks,
            increments=increments,
            n_classes=n_classes,
        )

    model = factory.get_model(config)
    model.inc_dataset = inc_dataset

    return inc_dataset, model


def _set_global_parameters(config):
    _set_seed(
        config["seed"],
        config["threads"],
        config["no_benchmark"],
        config["detect_anomaly"],
    )
    factory.set_device(config)


def _set_seed(seed, nb_threads, no_benchmark, detect_anomaly):
    logger.info("Set seed {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if no_benchmark:
        logger.warning("CUDA algos are not determinists but faster!")
    else:
        logger.warning("CUDA algos are determinists but very slow!")
    cudnn.benchmark = True  # This will slow down training.
    torch.set_num_threads(nb_threads)
    if detect_anomaly:
        logger.info("Will detect autograd anomaly.")
        torch.autograd.set_detect_anomaly(detect_anomaly)


def _set_up_options(args):
    options_paths = args["options"] or []

    autolabel = []
    for option_path in options_paths:
        if not os.path.exists(option_path):
            raise IOError("Not found options file {}.".format(option_path))

        args.update(_parse_options(option_path))

        autolabel.append(os.path.splitext(os.path.basename(option_path))[0])

    return "_".join(autolabel)


def _parse_options(path):
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.load(f, Loader=yaml.FullLoader)
        elif path.endswith(".json"):
            return json.load(f)["config"]
        else:
            raise Exception("Unknown file type {}.".format(path))


# ----
# Misc
# ----


def _aggregate_results(list_results):
    res = str(round(statistics.mean(list_results) * 100, 2))
    if len(list_results) > 1:
        res = res + " +/- " + str(round(statistics.stdev(list_results) * 100, 2))
    return res


def _log_metrics(metrics_results: dict, task_id: int):
    wandb_metrics = dict()
    wandb_metrics["task"] = task_id
    wandb_metrics["incremental_acc"] = metrics_results["incremental_accuracy"]
    wandb_metrics["average_acc"] = metrics_results["average_accuracy"]
    wandb_metrics["last_acc"] = metrics_results["accuracy"]["total"]
    wandb_metrics["incremental_acc_top5"] = metrics_results["incremental_accuracy_top5"]
    wandb_metrics["last_acc_top5"] = metrics_results["accuracy_top5"]["total"]
    wandb_metrics["forgetting"] = metrics_results["forgetting"]

    wandb.log(wandb_metrics)
