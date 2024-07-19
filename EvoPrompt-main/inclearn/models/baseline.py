from loguru import logger
from easydict import EasyDict as edict
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Union
import pickle
from typing import List, Dict
from collections import defaultdict
from timm import create_model
import wandb
from timm.optim import create_optimizer

import torch
from torchvision import transforms
from torch.nn import functional as F

from inclearn.lib import utils
from inclearn.lib.scaler import ContinualScaler
from inclearn.lib.utils import freeze_module
from inclearn.models.icarl import ICarl


class SimpleBaseline(ICarl):
    def __init__(self, args):
        args = edict(args)
        self._disable_progressbar = args.get("no_progressbar", False)
        self._device = args["device"][0]
        self._multiple_devices = args.device
        self.check_loss = args.get("check_loss", True)

        # task info
        self._classes_within_task = list()
        self._initial_increment = args.initial_increment
        self._increment = args.increment
        self._dataset_name = args.dataset
        self.num_classes_so_far = 0

        # training param
        self._start_epoch = args.get("start_epoch", "epochs")
        self._incremental_epochs = args.epochs
        self._fast_dev_run = args.fast_dev_run
        self.mixed_precision_training = args.get("mixed_precision_training", False)
        self._training_from_task = 0
        if self.mixed_precision_training:
            self.loss_scaler = ContinualScaler(self.mixed_precision_training)
        logger.info(f"Mixed precision training: {self.mixed_precision_training}")
        self._method = args.get("method")
        assert self._method in ["ft-nme", "ft-seq", "ft-lp"]
        logger.info(f"Method: {self._method}")

        # criterion
        self._cls_criterion = torch.nn.CrossEntropyLoss().to(self._device)
        print(f"Notes: {args.exp_notes}")

        if self._fast_dev_run:
            logger.info("Fast dev run mode")

        # init transforms
        self.domain_incremental = False
        normalize = False
        if self._dataset_name == "cifar100":
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        elif self._dataset_name == "cifar10":
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif self._dataset_name == "imagenet_r":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        elif self._dataset_name == "core50":
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            self.domain_incremental = True
            normalize = True
        else:
            raise ValueError(f"{self._dataset_name} is not supported")

        # train transforms
        train_transforms = [
            transforms.RandomResizedCrop(
                size=224, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]

        # test transforms
        if self._dataset_name in ["cifar10", "cifar100"]:
            test_transforms = [
                transforms.Resize(224, interpolation=3),
                transforms.ToTensor(),
            ]
        elif self._dataset_name in ["imagenet_r", "core50"]:
            test_transforms = [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]

        if normalize:
            train_transforms.append(transforms.Normalize(mean, std))
            test_transforms.append(transforms.Normalize(mean, std))

        self._train_transforms = transforms.Compose(train_transforms)
        self.test_transforms = transforms.Compose(test_transforms)

        # Optimization:
        self._batch_size = args.batch_size
        self._n_epochs = args.epochs
        self._optimizer_config = args.get("optimizer", None)
        self._lr_scheduler_config = args.get("lr_scheduler", None)
        self.grad_clip_norm = args.get("grad_clip_norm", 1.0)

        # Rehearsal Learning:
        self._build_examplars_every_x_epochs = False
        self._n_classes = self._initial_increment
        self._last_results = None
        self._validation_percent = args.validation

        self._evaluation_type = args.get("eval_type", "icarl")
        self._evaluation_config = args.get("evaluation_config", {})

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        # Model
        self._network = create_model(**args.convnet_config)
        self._set_network_trainable()
        self._network.to(self._device)
        self.prototypes = None
        self.prototypes_count = None

        self._memory_size = args.memory_size
        self.using_replay = self._memory_size != 0
        self._fixed_memory = args.get("fixed_memory", True)
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = self._initial_increment
        self._last_results = None
        self._validation_percent = args.validation

        self._examplars = {}
        self._means = None
        self._herding_indexes = []
        self._post_processing_type = None
        self._data_memory, self._targets_memory = None, None
        self._args = args
        self._args["_logs"] = {}
        self.results_folder = args["results_folder"]
        self.run_id = args["run_id"]

        self.dump_metadata = args.get("dump_metadata", None)

    def _set_network_trainable(self):
        if self._method in ["ft-nme", "ft-lp"]:
            freeze_module(self._network.patch_embed)
            freeze_module(self._network.pos_embed)
            freeze_module(self._network.blocks)
            freeze_module(self._network.cls_token)

        self.enable_training = True
        if self._method == "ft-nme":
            self.enable_training = False

    def _after_task_intensive(self, inc_dataset):
        pass

    def _after_task(self, inc_dataset):
        pass

    def get_memory(self):
        return None

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    def on_train_epoch_start(self):
        pass

    def masking_logits(
        self, logits: torch.Tensor, fill: Union[torch.Tensor, float] = float("-inf")
    ) -> None:
        num_model_classes = logits.size(1)
        mask = self._classes_within_task[self._task]
        not_cur_cls = np.setdiff1d(np.arange(num_model_classes), mask)
        not_cur_cls = torch.tensor(not_cur_cls, dtype=torch.int64).to(self._device)
        if len(not_cur_cls) > 0:
            if not self.domain_incremental:
                logits = logits.index_fill(dim=1, index=not_cur_cls, value=fill)

        return logits

    def _compute_loss(
        self,
        outputs,
        targets,
    ):
        logits = outputs
        logits = self.masking_logits(logits)
        loss = self._cls_criterion(logits, targets)
        self._metrics["cls"] += loss.item()
        wandb.log({"loss_step": loss.detach().cpu()})

        return loss

    def _forward_loss(
        self,
        training_network,
        inputs,
        targets,
        memory_flags,
        samples_idx,
        *args,
        **kwargs,
    ):

        if isinstance(inputs, list):
            inputs = torch.cat([inputs[0], inputs[1]], dim=0)
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(
            self._device, non_blocking=True
        )
        with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
            outputs = training_network(inputs)
            loss = self._compute_loss(
                outputs,
                targets,
            )

        if self.check_loss:
            if not utils.check_loss(loss):
                raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        return loss

    def _train_task(self, train_loader, val_loader):
        if self._method == "ft-nme":
            return

        logger.debug("nb {}.".format(len(train_loader.dataset)))
        clipper = None
        n_epochs = self._n_epochs
        if self._fast_dev_run:
            n_epochs = 1
        self._on_finetuning = False
        if self._task >= self._training_from_task:
            self._training_step(
                train_loader, val_loader, 0, n_epochs, record_bn=True, clipper=clipper
            )
        self._post_processing_type = None

    def _eval_task(self, test_loader):

        ypred = []
        ytrue = []

        if self._method == "ft-nme":
            last_epoch = True
        else:
            last_epoch = self._current_epoch == self.nb_epochs - 1
        if self.dump_metadata and last_epoch:
            out_metadata = defaultdict(list)

        assert self._network.training == False, "Model should be in eval model."

        if self._method == "ft-nme":
            prototypes = F.normalize(self.prototypes, dim=-1)
            if self.domain_incremental:
                prototypes = F.normalize(
                    self.prototypes / self.prototypes_count.unsqueeze(1), dim=-1
                )

        for input_dict in test_loader:
            ytrue.append(input_dict["targets"].numpy())

            with torch.no_grad():
                inputs = input_dict["inputs"].to(self._device)
                if self._method == "ft-nme":
                    logits_batch = self._predict_nme(inputs, prototypes)
                else:
                    logits_batch = self._predict_classifier(inputs)
                ypred.append(logits_batch.cpu().numpy())

                if last_epoch and self.dump_metadata:
                    feats = self._network.forward_features(inputs)[:, 0]
                    out_metadata["embeddings"].append(feats.detach().cpu())
                    out_metadata["logits"].append(logits_batch.detach().cpu())
                    out_metadata["labels"].append(input_dict["targets"].detach().cpu())
                    preds = F.softmax(logits_batch, dim=1)
                    out_metadata["preds"].append(preds.detach().cpu())

        ypred = np.concatenate(ypred, axis=0)
        ytrue = np.concatenate(ytrue)
        self._last_results = (ypred, ytrue)

        if last_epoch and self.dump_metadata:
            logger.info("Store Metadata: Embeddings, logits")
            logger.info(f"store metadata to {self.results_folder}")
            self.store_metadata(out_metadata)

        return ypred, ytrue

    def _predict_classifier(self, inputs: torch.Tensor):
        logits = self._network(inputs)
        logits = logits.detach()

        # masking not learned classes
        if not self.domain_incremental:
            mask = torch.arange(self.num_classes_so_far).to(self._device)
            logits_mask = torch.ones_like(logits, device=self._device) * float("-inf")
            logits_mask = logits_mask.index_fill(1, mask, 0.0)
            logits = logits + logits_mask

            logits[:, self.num_classes_so_far :] = -torch.inf

        preds = F.softmax(logits, dim=1)

        return preds

    def _predict_nme(self, inputs: torch.Tensor, prototypes: torch.Tensor):
        feats = self._network.forward_features(inputs)[:, 0]
        norm_feats = F.normalize(feats, dim=-1)

        logits_batch = list()
        for feat in norm_feats:
            distance = feat - prototypes.type_as(feat)
            logits = torch.linalg.norm(distance, ord=2, dim=1).unsqueeze(0)
            logits_batch.append(-logits)
        logits_batch = torch.cat(logits_batch, dim=0)

        return logits_batch

    def save_metadata(self, *args, **kwargs):
        pass

    def store_metadata(
        self,
        out_metadata: Dict[str, List[torch.Tensor]],
    ):
        for k in out_metadata.keys():
            out_metadata[k] = torch.cat(out_metadata[k], dim=0)

        output_dir = os.path.join(
            self.results_folder, "metadata_{}".format(self.run_id)
        )
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        filename1 = os.path.join(output_dir, f"metadata-{str(self._task)}.pkl")
        f1 = open((filename1), "wb+")
        pickle.dump(out_metadata, f1)

    def _configure_optimizer(self):
        if not self.enable_training:
            return

        # backbone optimizer
        params = list()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                params.append(param)
        params = [{"params": params}]
        self._optimizer = create_optimizer(self._optimizer_config, params)
        if self._lr_scheduler_config.method == "constant":
            self._scheduler = None

    def extract_prototypes(
        self,
        train_loader,
    ):
        if not self._method == "ft-nme":
            return

        with torch.no_grad():
            feats = list()
            labels = list()

            # using test transform to extract features
            dataloader = deepcopy(train_loader)
            dataloader.dataset.trsf = self.test_transforms

            for input_dict in tqdm(dataloader, desc="get prototypes"):
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                feat = self._network.forward_features(inputs)[:, 0].detach().cpu()
                feats.append(feat)
                labels.append(targets)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels)

        # get per class mean
        task_classes = sorted(self._classes_within_task[self._task])

        # domain incrementa prototypes
        if self.domain_incremental:
            task_classes = sorted(self._classes_within_task[0])
            class_sum = list()
            class_count = list()

            for class_idx in task_classes:
                cls_feats = feats[labels == class_idx]
                class_sum.append(cls_feats.sum(dim=0, keepdim=True))
                class_count.append(cls_feats.size(0))

            class_sum = torch.cat(class_sum, dim=0)
            class_count = torch.tensor(class_count)

            if self.prototypes == None and self.prototypes_count == None:
                self.prototypes_count = class_count
                self.prototypes = class_sum
            else:
                self.prototypes_count += class_count
                self.prototypes += class_sum

            return

        class_means = list()
        for class_idx in task_classes:
            cls_feats = feats[labels == class_idx]
            class_means.append(cls_feats.mean(dim=0, keepdim=True))
        class_means = torch.cat(class_means, dim=0)

        if self.prototypes == None:
            self.prototypes = class_means
        else:
            self.prototypes = torch.cat([self.prototypes, class_means], dim=0)

    def _before_task(self, train_loader, val_loader):
        if self._task == 0:
            for task_id in range(self._n_tasks):
                start = 0 if task_id == 0 else self._classes_within_task[-1][-1] + 1
                end = self._initial_increment if task_id == 0 else self._increment
                if self.domain_incremental:
                    start = 0
                    end = self._initial_increment
                cls = list(range(start, start + end))
                self._classes_within_task.append(cls)
        if not self.domain_incremental:
            self.num_classes_so_far += self._task_size
        elif self.domain_incremental and self._task == 0:
            self.num_classes_so_far += self._task_size

        # set epochs
        if self._task == 0:
            self._n_epochs = self._start_epoch
        else:
            self._n_epochs = self._incremental_epochs

        # network on task start
        self.extract_prototypes(train_loader)

        # reset optimizer and lr scheduler
        self._configure_optimizer()

        if self._task == 0:
            self._n_prev_classes = 0
        else:
            self._n_prev_classes = self._n_classes
            self._n_classes += self._task_size

        train_loader.dataset.trsf = self._train_transforms
        if self._dataset_name in [
            "cifar10",
            "cifar100",
            "imagenet_r",
            "core50",
        ]:
            val_loader.dataset.trsf = self.test_transforms
        else:
            raise RuntimeError(f"Dataset {self._dataset_name} is not supported.")

        # if self.use_query_rehearsal or self.use_reg:
        self.train_loader = deepcopy(train_loader)

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.grad_clip_norm)

    def save_parameters(self, directory, run_id):
        if self.mixed_precision_training:
            scaler_path = os.path.join(
                directory, f"scaler_{run_id}_task_{self._task}.pth"
            )
            logger.info(f"Saving scaler at {scaler_path}.")
            torch.save(self.loss_scaler.state_dict(), scaler_path)
        return super().save_parameters(directory, run_id)

    def load_parameters(self, directory, run_id):
        path = os.path.join(directory, f"net_{run_id}_task_{self._task}.pth")
        if not os.path.exists(path):
            return False

        ckpt = torch.load(path)
        logger.info(f"Loading model at {path}.")
        self.network.load_state_dict(ckpt, strict=False)

        if self.mixed_precision_training:
            scaler_path = os.path.join(
                directory, f"scaler_{run_id}_task_{self._task}.pth"
            )
            if not os.path.exists(scaler_path):
                logger.info(f"Cannot find scaler checkpoint on {scaler_path}")
            else:
                scaler_ckpt = torch.load(scaler_path)
                self.loss_scaler.load_state_dict(scaler_ckpt)
                logger.info(f"Successfully loading scaler at {scaler_path}.")

        return True

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, model_path: str):
        logger.info(f"Loading model at {model_path}.")
        try:
            self.network.load_state_dict(
                torch.load(model_path, map_location="cpu"), strict=False
            )
        except Exception:
            logger.warning("Old method to save weights, it's deprecated!")
            self._network = torch.load(model_path)

    def backward(self, loss: torch.Tensor):
        if self.mixed_precision_training:
            self.loss_scaler.pre_step(
                loss=loss,
                optimizer=self._optimizer,
                parameters=[
                    param for param in self._network.parameters() if param.requires_grad
                ],
            )
        else:
            super().backward(loss)
