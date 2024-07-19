from operator import itemgetter
from loguru import logger
from easydict import EasyDict as edict
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from typing import Union, Dict
import pickle
from typing import List
from collections import defaultdict
import wandb

import torch
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from timm.optim import create_optimizer
import captum

from inclearn.lib import network, utils
from inclearn.lib.scaler import ContinualScaler
from inclearn.models.icarl import ICarl
from inclearn.lib.utils import unfreeze_module
from inclearn.lib.clustering import KMeans
from inclearn.lib.optimal_transport import get_wassersteinized_layers_modularized


class EvoPrompt(ICarl):
    """Prompt-based Continual Learning."""

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
        self._num_workers = args["workers"]
        self._start_epoch = args.get("start_epoch", "epochs")
        self._incremental_epochs = args.epochs
        self._fast_dev_run = args.fast_dev_run
        self.mixed_precision_training = args.get("mixed_precision_training", False)
        if self.mixed_precision_training:
            self.loss_scaler = ContinualScaler(self.mixed_precision_training)
        logger.info(f"Mixed precision training: {self.mixed_precision_training}")

        # criterion
        self._cls_criterion = torch.nn.CrossEntropyLoss().to(self._device)
        print(f"Notes: {args.exp_notes}")

        if self._fast_dev_run:
            logger.info("Fast dev run mode")

        # init transforms
        self.domain_incremental = False
        self.dil_summary = args.get("dil_summary", "mean")
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
        self._optimizer_config = args.optimizer
        self._lr_scheduler_config = args.lr_scheduler

        # Rehearsal Learning:
        self._build_examplars_every_x_epochs = args.get(
            "build_examplars_every_x_epochs", True
        )
        self._memory_size = args.memory_size
        self.using_replay = self._memory_size != 0
        self._fixed_memory = args.get("fixed_memory", True)
        self._herding_selection = args.get("herding_selection", {"type": "icarl"})
        self._n_classes = self._initial_increment
        self._last_results = None
        self._validation_percent = args.validation

        self._evaluation_type = args.get("eval_type", "icarl")
        self._evaluation_config = args.get("evaluation_config", {})

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._early_stopping = args.get("early_stopping", {})

        # Model
        self._classifier_config = args.classifier_config
        model_kwargs = args.model_kwargs
        # self._classifier_config['num_classes'] = self._initial_increment
        self.prompt_config = args.get("prompt_config", None)
        self._network = network.ViTPrompt(
            convnet_kwargs=args.convnet_config,
            classifier_name=args.classifier_name,
            classifier_kwargs=self._classifier_config,
            prompt_kwargs=self.prompt_config,
            device=self._device,
            initial_increment=self._initial_increment,
            **model_kwargs,
        )

        self._examplars = {}
        self._means = None
        self._herding_indexes = []
        self._post_processing_type = None
        self._data_memory, self._targets_memory = None, None
        self._args = args
        self._args["_logs"] = {}
        self.results_folder = args["results_folder"]
        self.run_id = args["run_id"]
        self.class_means = list()
        self.use_nce = False

        # finetuning
        self._finetuning_config = args.get("finetuning_config", None)

        # training strategy
        self.use_train_mask = args.use_train_mask
        self.grad_clip_norm = args.grad_clip_norm

        # meta-learn the memory
        self.incremental_fusion = args.get("incremental_fusion", None)
        self.use_incremental_fusion = (
            True if self._network.prompt_memory != None else False
        )
        self.use_prompt = True if self._network.prompt_memory else False
        self.compositional_initialization = args.get(
            "compositional_initialization", None
        )

        # attribution
        self.attribution_aware_fusion = args.get("attribution_aware_fusion", None)
        self.current_task_attribution = None
        self.global_attribution = None
        self._store_weights = args.get("save_model", False)
        self._weights_path = "weights/ot_best"
        self._use_ot = args.get("use_optimal_transport_alignment", None)
        self._training_from_task = 0
        self._log_past_task_accuracy = args.get(f"log_past_task_accuracy", False)

        # compute storage
        self._compute_storage()

    def _compute_storage(self) -> None:
        # FFN prompt memory
        num_storage = sum(p.numel() for p in self.network.parameters())
        total_additional_storage = 0

        # memory fusion: storage for global memory
        if self.network.prompt_memory != None and self.use_incremental_fusion:
            num_global_memory = sum(
                p.numel() for p in self.network.prompt_memory.parameters()
            )
            logger.info(f"num_global_memory: {num_global_memory:,}")
            num_storage += num_global_memory
            total_additional_storage += num_global_memory * 2

        # memory fusion: storage for node attribution
        if self.network.prompt_memory != None and self.use_incremental_fusion:
            num_prompt_layers = len(self.network.prompt_layer_idx)
            hidden_nodes_attr = (
                self.network.prompt_memory.hidden_dim * num_prompt_layers
            )
            out_nodes_attr = (
                self.network.prompt_memory.embed_dim
                * self.network.prompt_memory.length
                * num_prompt_layers
            )
            num_node_attribution = hidden_nodes_attr + out_nodes_attr
            logger.info(f"num_node_attribution: {num_node_attribution:,}")
            num_storage += num_node_attribution
            total_additional_storage += num_node_attribution

        # classifier transfer: storage for prototypes
        if self.compositional_initialization:
            tensor_prototypes = sum(
                p.numel() for p in self.network.classifier.parameters()
            )
            logger.info(f"tensor_prototypes: {tensor_prototypes:,}")
            num_storage += tensor_prototypes
            total_additional_storage += tensor_prototypes

        logger.info(f"Total storage: {num_storage:,}")
        logger.info(f"Total additional storage: {total_additional_storage:,}")

    def get_prompt_memory_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for param in list(self._network.prompt_memory.parameters()):
            params.append(param.view(-1))
        return torch.cat(params)

    def set_memory_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_prompt_memory_params().size()
        progress = 0
        for pp in list(self._network.prompt_memory.parameters()):
            cand_params = new_params[
                progress : progress + torch.tensor(pp.size()).prod()
            ].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_memory_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self._network.prompt_memory.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    @property
    def _memory_per_class(self):
        """Returns the number of examplars per class."""
        if self._fixed_memory:
            return self._memory_size // self._total_n_classes
        return self._memory_size // self._n_classes

    def _train_task(self, train_loader, val_loader):
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

    @property
    def weight_decay(self):
        if isinstance(self._weight_decay, float):
            return self._weight_decay
        elif isinstance(self._weight_decay, dict):
            start, end = self._weight_decay["start"], self._weight_decay["end"]
            step = (max(start, end) - min(start, end)) / (self._n_tasks - 1)
            factor = -1 if start > end else 1

            return start + factor * self._task * step
        raise TypeError(
            "Invalid type {} for weight decay: {}.".format(
                type(self._weight_decay), self._weight_decay
            )
        )

    def on_train_epoch_start(self):
        pass

    def _after_task(self, inc_dataset):
        self._network.on_task_end()

        if self._training_from_task > self._task:
            path = os.path.join(
                self._weights_path, f"model_before_agg_{self._task}.pth"
            )
            ckpt = torch.load(path, self._device)
            logger.info(f"Loading model at {path}.")
            self.network.load_state_dict(ckpt)

            path2 = os.path.join(self._weights_path, f"prompt_memory_{self._task}.pth")
            ckpt2 = torch.load(path2, self._device)
            logger.info(f"Loading prompt memory at {path2}")
            self.prev_reference_prompt_memory = ckpt2
            self._current_epoch = self._n_epochs
            self.nb_epochs = self._n_epochs

        if self._store_weights:
            logger.info(f"store weights on {self._weights_path}")
            # store params before aggregation (adapted prompt)
            os.makedirs(self.results_folder, exist_ok=True)
            path = os.path.join(
                self.results_folder, f"working_pm-before_alignment-{self._task}.pth"
            )
            torch.save(self.network.prompt_memory.state_dict(), path)
            # store stable consolidated prompt memory
            path2 = os.path.join(self.results_folder, f"reference_pm-{self._task}.pth")
            torch.save(self.prev_reference_prompt_memory, path2)

        if self._task > 0 and self._use_ot:
            self._compute_transport_matrix()
            if self._store_weights:
                path = os.path.join(
                    self.results_folder, f"working_pm-after_alignment-{self._task}.pth"
                )
                torch.save(self.aligned_new_working_memory, path)
        if self.attribution_aware_fusion:
            self._compute_importance()

        if (
            self.use_incremental_fusion
            and self.incremental_fusion.across_task_meta_update
            and self._task > 0
        ):
            self._aggregate_params()
            logger.info("Set memory params on task end.")

        # fuse importance
        if self.attribution_aware_fusion:
            if self.global_attribution is None:
                self.global_attribution = self.current_task_attribution
            elif (
                self.global_attribution != None
                and self.current_task_attribution != None
            ):
                self._fuse_attribution()

    def _fuse_attribution(self):
        new_global_attribution = list()
        for i in range(len(self.global_attribution)):
            current_task_attribution = self.current_task_attribution[i].unsqueeze(2)
            global_attribution = self.global_attribution[i].unsqueeze(2)
            fused_attribution = (
                torch.cat([global_attribution, current_task_attribution], dim=2)
                .max(dim=2)
                .values
            )
            new_global_attribution.append(fused_attribution)
        self.global_attribution = new_global_attribution

    def _compute_transport_matrix(self):
        activation_based = False
        kwargs = edict()

        if activation_based:
            train_loader = self._get_train_loader(batch_size=64)
            kwargs.dataloader = train_loader
            kwargs.base_model = self

        new_working_prompt = self._network.prompt_memory
        old_prompt_reference_prompt = deepcopy(self._network.prompt_memory)
        old_reference_prompt_params = self.prev_reference_prompt_memory
        progress = 0

        # put params into old prompt module
        for pp in list(old_prompt_reference_prompt.parameters()):
            cand_params = old_reference_prompt_params[
                progress : progress + torch.tensor(pp.size()).prod()
            ].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

        aligned_new_working_memory, cur_T_vars = get_wassersteinized_layers_modularized(
            models=[new_working_prompt, old_prompt_reference_prompt],
            activation_based=activation_based,
            **kwargs,
        )

        self.aligned_new_working_memory = aligned_new_working_memory
        self.cur_T_vars = cur_T_vars

    def _get_current_attribution(self, normalize: bool = False):
        if self.current_task_attribution is None:
            return None

        if not normalize:
            return torch.cat([attr.flatten() for attr in self.current_task_attribution])

        new_cur_attr = list()
        for i, attr in enumerate(self.current_task_attribution):
            attr = attr.unsqueeze(2)
            denominator = (
                1e-9 if (attr.max() - attr.min()) == 0 else (attr.max() - attr.min())
            )
            attr = (attr - attr.min()) / denominator
            new_cur_attr.append(attr)

        return torch.cat([attr.flatten() for attr in new_cur_attr])

    def _get_global_attribution(self, normalize: bool = False):
        if self.global_attribution is None:
            return None

        if not normalize:
            return torch.cat([attr.flatten() for attr in self.global_attribution])

        new_global_attr = list()
        for i, attr in enumerate(self.global_attribution):
            attr = attr.unsqueeze(2)
            denominator = (
                1e-9 if (attr.max() - attr.min()) == 0 else (attr.max() - attr.min())
            )
            attr = (attr - attr.min()) / denominator
            new_global_attr.append(attr)

        return torch.cat([attr.flatten() for attr in new_global_attr])

    def _compute_importance(self):
        logger.info("Compute prompt importance")
        self._network.eval()
        modules = self._network.get_memory_modules()
        hidden_dim = modules[0].hidden_features
        input_dim = modules[0].out_features
        attribution_method = captum.attr.LayerActivation
        single_inference = attribution_method in [
            captum.attr.LayerGradientXActivation,
            captum.attr.LayerActivation,
        ]

        def forward_func(
            inputs: torch.Tensor, targets: torch.Tensor, method: EvoPrompt
        ):
            # get prompt
            with torch.no_grad():
                query_embed = self._network.forward_query(
                    inputs,
                )
            logits = self._network(inputs, query_embed=query_embed)["logits"]

            method.logits_attribution = logits

            return logits

        # get list of hooked modules
        hooked_modules = list()
        for m in modules:
            hooked_modules.append(m.block[1])  # hidden after relu
            hooked_modules.append(m.block[2])  # output dim

        if single_inference:
            hooked_modules = [hooked_modules]

        attribution = list()
        for k, hooked_module in enumerate(hooked_modules):
            logger.info(f"compute attribution of module {k + 1}/{len(hooked_modules)}")
            # create attribution for selected module
            print(hooked_module)
            attr_method = attribution_method(
                forward_func=forward_func, layer=hooked_module
            )
            attr_out = self._run_attribution(attr_method)

            if isinstance(attr_out, torch.Tensor):
                attribution.append(attr_out)
            else:
                attribution = attr_out

        attribution_per_params = list()
        for attr in attribution:
            inp_dim = input_dim if attr.size(0) == hidden_dim else hidden_dim
            attr = attr.unsqueeze(1).repeat(1, inp_dim)
            attribution_per_params.append(attr)

        self.current_task_attribution = attribution_per_params

    def _run_attribution(self, attr_method: captum.attr.Attribution):
        dataloader = self._get_train_loader(batch_size=64)
        iterator = tqdm(dataloader, f"compute attribution (module level)")

        attribution = list()
        for data in iterator:
            inputs, targets = data["inputs"].to(self._device), data["targets"].to(
                self._device
            )
            with torch.no_grad():
                attr_out = attr_method.attribute((inputs, targets, self))
            attr_out = (
                attr_out.detach().cpu()
                if isinstance(attr_out, torch.Tensor)
                else [attr.detach().cpu() for attr in attr_out]
            )
            preds = F.softmax(self.logits_attribution, dim=-1).max(dim=-1).indices
            attr_out = [F.relu(attr[preds.cpu() == targets.cpu()]) for attr in attr_out]
            attribution.append(attr_out)

        if isinstance(attribution[0], torch.Tensor):
            attribution_results = torch.cat(attribution, dim=0).mean(0)  # bsz, ndim
        else:
            # concat for each module along batch dimension
            attribution_results = list()
            num_modules = len(attribution[0])
            for module_idx in range(num_modules):
                attribution_per_module = list()
                num_batch = len(attribution)
                for batch_idx in range(num_batch):
                    attribution_per_module.append(attribution[batch_idx][module_idx])
                attribution_per_module = torch.cat(
                    attribution_per_module, dim=0
                )  # num_samples, ndim
                print(
                    f"module attribution {module_idx} shape: {attribution_per_module.shape}"
                )
                attribution_per_module = attribution_per_module.mean(dim=0)
                print(
                    f"Attr mean: {attribution_per_module.mean()}, max: {attribution_per_module.max()}, min: {attribution_per_module.min()}"
                )
                attribution_results.append(attribution_per_module)
        return attribution_results

    def _align_attribution(self):
        if self._use_ot and self.current_task_attribution is not None:
            aligned_attribution = list()
            num_iter = len(self.current_task_attribution)
            for idx in range(num_iter):
                T_var = self.cur_T_vars[idx]
                attr = self.current_task_attribution[idx]
                aligned_attr = torch.matmul(T_var.t().cpu(), attr)
                aligned_attribution.append(aligned_attr)
            self.current_task_attribution = aligned_attribution

    def _aggregate_params(self):
        aligned_new_working_memory = (
            self.aligned_new_working_memory
            if self._use_ot
            else self.get_prompt_memory_params()
        )
        if self._use_ot and self.attribution_aware_fusion:
            self._align_attribution()

        if self.attribution_aware_fusion:
            global_importance = self._get_global_attribution(normalize=True).type_as(
                self.prev_reference_prompt_memory
            )
            current_importance = self._get_current_attribution(normalize=True).type_as(
                self.prev_reference_prompt_memory
            )

        # fusion weight
        alpha = self.incremental_fusion.alpha
        weight = torch.ones_like(self.prev_reference_prompt_memory) * alpha
        if self.attribution_aware_fusion:
            weight = (torch.ones_like(self.prev_reference_prompt_memory) * alpha) + (
                (current_importance - global_importance) * alpha
            )
        print(
            f"Adaptation weight mean: {weight.mean()}, max: {weight.max()}, min: {weight.min()}"
        )

        new_params = self.prev_reference_prompt_memory + weight * (
            aligned_new_working_memory - self.prev_reference_prompt_memory
        )
        self.set_memory_params(new_params)

    def _get_train_loader(
        self, shuffle: bool = True, batch_size: int = None, rasio: float = None
    ):
        batch_size = self._batch_size if batch_size is None else batch_size
        train_dataset = deepcopy(self.train_loader.dataset)
        if rasio is not None:
            len_dataset = len(train_dataset)
            perm_indices = torch.randperm(len_dataset)
            num_samples = max(int(rasio * len_dataset), 1)
            indices = perm_indices[:num_samples]
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
        train_dataset.trsf = self.test_transforms
        dataloader = DataLoader(
            train_dataset, batch_size, shuffle, num_workers=self._num_workers
        )

        return dataloader

    def _after_task_intensive(self, inc_dataset):
        if self.using_replay:
            super()._after_task_intensive(inc_dataset)

    def get_memory(self):
        if self.using_replay:
            return self._data_memory, self._targets_memory
        else:
            return None

    def compute_class_prototypes(self, num_prototypes: int = 1):
        train_dataset = deepcopy(self.train_loader.dataset)
        train_dataset.trsf = self.test_transforms
        dataloader = DataLoader(
            train_dataset,
            self._batch_size,
            False,
            num_workers=self.train_loader.num_workers,
        )
        out = self.extract_features(
            dataloader,
            get_deep_features=False,
            original_inference=False,
            use_prompt=True,
        )
        feats, labels = out.embedding[: out.labels.size(0)], out.labels

        # get per class mean
        task_classes = sorted(self._classes_within_task[self._task])
        class_means = list()
        for class_idx in tqdm(task_classes, "computer class means"):
            cls_feats = feats[labels == class_idx]
            if num_prototypes == 1:
                class_means.append(cls_feats.mean(dim=0, keepdim=True))
            else:
                _, centroids = KMeans(cls_feats, device=self._device, K=num_prototypes)
                class_means.append(centroids)

        return torch.cat(class_means, dim=0)

    def collect_class_prototypes(self, num_prototypes: int = 1):
        class_means = self.compute_class_prototypes(num_prototypes=num_prototypes)
        try:
            self.class_means[self._task] = class_means
        except:
            self.class_means.append(class_means)

    def _eval_task(self, test_loader):
        kwargs = edict()
        if self._evaluation_type in ("icarl", "nme"):
            return super()._eval_task(test_loader)
        elif self._evaluation_type in ("softmax", "cnn"):
            if self.use_nce:
                self.collect_class_prototypes()
                kwargs.prototypes = torch.cat(self.class_means, dim=0)

            ypred = []
            ytrue = []
            last_epoch = self._current_epoch == self.nb_epochs - 1

            assert self._network.training == False, "Model should be in eval model."
            for input_dict in test_loader:
                ytrue.append(input_dict["targets"].numpy())

                with torch.no_grad():
                    inputs = input_dict["inputs"].to(self._device)
                    if self.use_prompt:
                        kwargs.query_embed = self._network.forward_query(inputs)
                    if last_epoch:
                        logits, feats, attn = itemgetter("logits", "feats", "attn")(
                            self._network(inputs, **kwargs)
                        )
                    else:
                        logits = itemgetter("logits")(self._network(inputs, **kwargs))
                    logits = logits.detach()
                    if not self.domain_incremental:
                        mask = torch.arange(self.num_classes_so_far).to(self._device)
                        logits_mask = torch.ones_like(
                            logits, device=self._device
                        ) * float("-inf")
                        logits_mask = logits_mask.index_fill(1, mask, 0.0)
                        logits = logits + logits_mask

                        logits[:, self.num_classes_so_far :] = -torch.inf
                    elif self.domain_incremental:
                        B, NC = logits.shape
                        logits = logits.reshape(
                            B, NC // self._task_size, self._task_size
                        )
                        logits = logits[:, : self._task + 1]
                        if self.dil_summary == "max":
                            logits = torch.max(logits, dim=1).values
                        elif self.dil_summary == "mean":
                            logits = torch.mean(logits, dim=1)
                        else:
                            raise Exception(f"Error")

                    preds = F.softmax(logits, dim=1)
                    ypred.append(preds.cpu().numpy())

            ypred = np.concatenate(ypred)
            ytrue = np.concatenate(ytrue)
            self._last_results = (ypred, ytrue)

            return ypred, ytrue
        else:
            raise ValueError(self._evaluation_type)

    def _register_hook(self):
        prompt_memory = self.network.get_memory_modules()
        self.prompt_activation = [None] * len(prompt_memory)
        self.hooks = list()

        def getActivation(idx):
            # the hook signature
            def hook(model, input, output):
                self.prompt_activation[idx] = output.detach().cpu()

            return hook

        # get list of hooked modules
        for l, m in enumerate(prompt_memory):
            hook = m.block[1].register_forward_hook(
                getActivation(l)
            )  # hidden after relu
            self.hooks.append(hook)

    def _remove_hook(self):
        for i, _ in enumerate(self.hooks):
            self.hooks[i].remove()
            self.hooks[i] = None
        self.prompt_activation = [None]

    def save_metadata(self, *args, **kwargs):
        pass

    def _configure_optimizer(self):
        # backbone optimizer
        params = list()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                params.append(param)
        params = [{"params": params}]
        self._optimizer = create_optimizer(self._optimizer_config, params)
        if self._lr_scheduler_config.method == "constant":
            self._scheduler = None
        elif self._lr_scheduler_config.method == "cosine":
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=self._n_epochs,
                eta_min=self._lr_scheduler_config.eta_min,
                verbose=self._lr_scheduler_config.get("verbose", False),
            )
        elif self._lr_scheduler_config.method == "exponential":
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer,
                self._lr_scheduler_config.gamma,
                verbose=self._lr_scheduler_config.get("verbose", False),
            )

    def extract_features(
        self,
        train_loader,
        get_deep_features: bool = False,
        use_prompt: bool = False,
        original_inference: bool = True,
    ):
        with torch.no_grad():
            out = edict()
            embedding = list()
            datapoint_index = list()
            labels = list()
            if get_deep_features:
                multiscale_feats = list()

            # using test transform to extract features
            dataloader = deepcopy(train_loader)
            dataloader.dataset.trsf = self.test_transforms

            for input_dict in tqdm(dataloader, desc="extract feats"):
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                samples_idx = input_dict["idx"]
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                query_embed = None
                if use_prompt:
                    query_embed = self._network.forward_query(inputs)
                    original_inference = False

                out = self._network.extract(
                    inputs,
                    original_inference=original_inference,
                    get_deep_features=get_deep_features,
                    query_embed=query_embed,
                )

                embedding.append(out.embedding.cpu())
                labels.append(targets)
                datapoint_index.append(samples_idx)
                if get_deep_features:
                    multiscale_feats.append(out.deep_feats.cpu())

        out.embedding = torch.cat(embedding, dim=0)
        out.labels = torch.cat(labels)
        out.datapoint_index = torch.cat(datapoint_index)
        if get_deep_features:
            out.multiscale_feats = torch.cat(multiscale_feats, dim=1)

        return out

    def _generate_list_classes(self):
        for task_id in range(self._n_tasks):
            start = 0 if task_id == 0 else self._classes_within_task[-1][-1] + 1
            end = self._initial_increment if task_id == 0 else self._increment
            if self.domain_incremental:
                start = 0 if task_id == 0 else self._classes_within_task[-1][-1] + 1
                end = self._n_classes
            cls = list(range(start, start + end))
            self._classes_within_task.append(cls)

    def _before_task(self, train_loader, val_loader):
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        if self._task == 0:
            self._generate_list_classes()
            if self.domain_incremental:
                self.num_classes = self._n_classes

        if not self.domain_incremental:
            self.num_classes_so_far += self._task_size
        elif self.domain_incremental:
            self.num_classes_so_far += self.num_classes
            self._task_size = self.num_classes

        # set epochs
        if self._task == 0:
            self._n_epochs = self._start_epoch
        else:
            self._n_epochs = self._incremental_epochs

        # network on task start
        kwargs = edict()
        if self.use_prompt or self.compositional_initialization:
            extract_out = self.extract_features(train_loader)
            kwargs.update(extract_out)
            kwargs.task_classes = self._classes_within_task[self._task]

        self._network.on_task_start(
            task_id=self._task,
            num_classes=self._task_size,
            num_prev_classes=(
                0 if self._task == 0 else len(self._classes_within_task[self._task - 1])
            ),
            **kwargs,
        )

        if self._task > 0:
            self._network.add_classes(self._task_size)

        if self.compositional_initialization:
            if self.domain_incremental and self._task >= 1:
                extract_out.labels += self.num_classes_so_far - self._task_size
            self._compositional_initialization(
                extract_out.embedding, extract_out.labels.cpu()
            )

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
        self.train_loader = deepcopy(train_loader)

        if (
            self.use_incremental_fusion
            and self.incremental_fusion.across_task_meta_update
        ):
            self._clone_params()

    def _compositional_initialization(
        self, embedding: torch.Tensor, labels: torch.Tensor
    ):
        # get class prototypes from pre-trained
        unique_classes = torch.unique(labels)
        mean_embeddings = list()
        for c in unique_classes:
            cls_embedding = embedding[labels == c]
            mean_embedding = cls_embedding.mean(0, keepdim=True)
            mean_embeddings.append(mean_embedding)
        mean_embeddings = torch.cat(mean_embeddings, dim=0)

        self.class_prototypes = (
            mean_embeddings
            if self._task == 0
            else torch.cat([self.class_prototypes, mean_embeddings], dim=0)
        )

        if self._task > 0:
            logger.info("Classifier transfer before task")
            classifier = (
                self._network.classifier.weight.data
                if isinstance(self._network.classifier, torch.nn.Linear)
                else self._network.classifier._weights0.data
            )

            num_classes_so_far = self.num_classes_so_far - self._task_size
            print("num_classes_so_far", num_classes_so_far)
            key = self.class_prototypes[:num_classes_so_far].to(self._device)
            query = mean_embeddings.type_as(key)

            num_heads = self.compositional_initialization.num_heads
            ndim = query.size(1)
            query = query.reshape(
                self._task_size, num_heads, ndim // num_heads
            ).permute(
                1, 0, 2
            )  # cur_classes, num_heads, ndim -> num_heads, cur_classes, ndim
            key = key.reshape(num_classes_so_far, num_heads, ndim // num_heads).permute(
                1, 0, 2
            )  # prev_classes, num_heads, ndim -> num_heads, prev_classes, ndim

            query_norm = F.normalize(query, dim=-1)
            key_norm = F.normalize(key, dim=-1)

            attn = query_norm @ key_norm.permute(
                0, 2, 1
            )  # num_heads, cur_classes, prev_classes
            attn = F.softmax(
                attn / self.compositional_initialization.temp, dim=-1
            )  # num_heads, cur_classes, prev_classes
            print(f"attn shape: {attn.shape}")
            print(f"attn max: {attn.max(-1).values}")
            print(f"attn min: {attn.min(-1).values}")
            print(f"attn mean: {attn.mean(-1)}")

            classifier = classifier[:num_classes_so_far].detach()
            classifier = classifier.reshape(
                num_classes_so_far, num_heads, ndim // num_heads
            ).permute(1, 0, 2)
            new_classifier = attn @ classifier  # num_heads, cur_classes, ndim
            new_classifier = new_classifier.permute(1, 0, 2).reshape(
                self._task_size, ndim
            )
            sorted_current_classes = unique_classes.sort().values
            print("sorted_current_classes", sorted_current_classes)
            print("new_classifier.shape", new_classifier.shape)

            if isinstance(self._network.classifier, torch.nn.Linear):
                self._network.classifier.weight.data[sorted_current_classes] = (
                    new_classifier
                )
            else:
                self._network.classifier._weights0.data[sorted_current_classes] = (
                    new_classifier
                )

    def _clone_params(self):
        self.prev_reference_prompt_memory = self.get_prompt_memory_params().data.clone()

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(self._network.parameters(), self.grad_clip_norm)

    def masking_logits(
        self, logits: torch.Tensor, fill: Union[torch.Tensor, float] = float("-inf")
    ) -> None:
        num_model_classes = logits.size(1)
        mask = self._classes_within_task[self._task]
        not_cur_cls = np.setdiff1d(np.arange(num_model_classes), mask)
        not_cur_cls = torch.tensor(not_cur_cls, dtype=torch.int64).to(self._device)
        if len(not_cur_cls) > 0:
            if True:
                logits = logits.index_fill(dim=1, index=not_cur_cls, value=fill)

        return logits

    def _compute_loss(
        self,
        outputs,
        targets,
    ):
        logits = outputs.logits
        if self.use_train_mask:
            logits = self.masking_logits(logits)

        if self.domain_incremental and self._task >= 1:
            targets += self.num_classes_so_far - self._task_size

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
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)
        bsz = targets.size(0)

        network_kwargs = edict()
        if self.use_prompt:
            with torch.no_grad():
                network_kwargs.query_embed = self._network.forward_query(inputs)

        y_query = None
        with torch.cuda.amp.autocast(enabled=self.mixed_precision_training):
            outputs = training_network(inputs, **network_kwargs)
            loss = self._compute_loss(
                outputs,
                targets,
            )

        if self.check_loss:
            if not utils.check_loss(loss):
                raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        return loss

    def generate_tasks_masks(self, targets: torch.Tensor):
        bsz = len(targets)
        tasks_masks = torch.zeros(bsz, 1, dtype=torch.int64).to(self._device)  # bsz, 1
        expanded_targets = targets.unsqueeze(1)
        for task_id, classes_task_i in enumerate(self._classes_within_task):
            task_i_idx = (
                expanded_targets
                == torch.tensor(classes_task_i)
                .type_as(expanded_targets)
                .unsqueeze(0)
                .repeat(bsz, 1)
            ).any(dim=1)
            tasks_masks[task_i_idx] = task_id

        return tasks_masks

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

    def optimizer_step(self):
        if self.mixed_precision_training:
            self.loss_scaler.post_step(self._optimizer, self._network)
        else:
            super().optimizer_step()

    def on_train_epoch_end(self, prog_bar, training_network):
        pass
