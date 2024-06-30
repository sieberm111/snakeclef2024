import os
import logging
import torch
import wandb
from typing import Tuple

import pandas as pd
import torch.nn as nn

from scipy.special import softmax
from torch.utils.data import DataLoader

from fgvc.core.training import train, predict
from fgvc.datasets import get_dataloaders
from fgvc.losses import FocalLossWithLogits, SeesawLossWithLogits
from fgvc.utils.experiment import (
    get_optimizer_and_scheduler,
    load_args,
    load_config,
    load_model,
    load_train_metadata,
    save_config,
)
from fgvc.utils.utils import set_cuda_device, set_random_seed
from fgvc.utils.wandb import (
    finish_wandb,
    init_wandb,
    resume_wandb,
    set_best_scores_in_summary,
)

logger = logging.getLogger("script")

class CombinedLoss(nn.Module):
    def __init__(self, class_count, class_df, batch_size=32):
        super(CombinedLoss, self).__init__()
        self.seesaw_loss = SeesawLossWithLogits(class_counts=class_count)
        self.class_dict = class_df.set_index("class_id").to_dict()['count']
        self.class_total = sum(self.class_dict.values())
        self.batch_size = batch_size

    def forward(self, outputs, targets):
        batch_total = sum([self.class_dict[x] for x in targets.cpu().numpy()])
        class_boost = (self.class_total*self.batch_size) / batch_total
        loss = self.seesaw_loss(outputs, targets)
        return loss*class_boost

def load_metadata(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load metadata of the traning and validation sets."""
    assert "dataset" in config

    BASE_IMAGE_PATH_iNAT = "train/SnakeCLEF2023-small_size"
    # BASE_IMAGE_PATH_HMP = "train/"

    train_metadata_iNat = pd.read_csv("./metadata/SnakeCLEF2023-Train-iNat-small-cleared.csv")
    # train_metadata_hmp = pd.read_csv("./metadata/SnakeCLEF2023-TrainMetadata-HM.csv")
    val_metadata = pd.read_csv("./metadata/SnakeCLEF2023-ValMetadata.csv")
    venomous_metadata = pd.read_csv("./metadata/venomous_status_list.csv")

    train_metadata_iNat["image_path"] = train_metadata_iNat.image_path.apply(
        lambda path: os.path.join(BASE_IMAGE_PATH_iNAT, path)
    )

    val_metadata["image_path"] = val_metadata.image_path.apply(
        lambda path: os.path.join(BASE_IMAGE_PATH_iNAT, path)
    )

    # train_metadata_hmp["image_path"] = train_metadata_hmp.image_path.apply(
    #     lambda path: os.path.join(BASE_IMAGE_PATH_HMP, path)
    # )

    metadata = pd.concat(
        [train_metadata_iNat, val_metadata]).reset_index(drop=True)  # train_metadata_hmp

    metadata_o = pd.merge(metadata, venomous_metadata, left_on='class_id', right_on='class_id', how='left')

    train_df = metadata_o[metadata_o.subset != "val"]
    valid_df = metadata_o[metadata_o.subset == "val"]

    species_df = pd.read_csv("./metadata/class_count.csv")

    return train_df, valid_df, species_df


def add_metadata_info_to_config(
        config: dict, train_df: pd.DataFrame, valid_df: pd.DataFrame
) -> dict:
    """Include information from metadata to the training configuration."""
    assert "class_id" in train_df and "class_id" in valid_df
    config["number_of_classes"] = len(train_df.class_id.unique())
    config["training_samples"] = len(train_df)
    config["test_samples"] = len(valid_df)
    return config


def train_clf(
        *,
        train_metadata: str = None,
        valid_metadata: str = None,
        config_path: str = None,
        cuda_devices: str = None,
        wandb_entity: str = None,
        wandb_project: str = None,
        resume_exp_name: str = None,
        **kwargs,
):
    """Train model on the classification task."""
    if train_metadata is None or valid_metadata is None or config_path is None:
        # load script args
        args, extra_args = load_args()
        config_path = args.config_path
        cuda_devices = args.cuda_devices
        wandb_entity = args.wandb_entity
        wandb_project = args.wandb_project
        resume_exp_name = args.resume_exp_name
    else:
        extra_args = kwargs

    # load training config
    logger.info("Loading training config.")
    config = load_config(
        config_path,
        extra_args,
        run_name_fmt="architecture-loss-augmentations",
        resume_exp_name=resume_exp_name,
    )

    # set device and random seed
    device = set_cuda_device(cuda_devices)
    set_random_seed(config["random_seed"])

    # load metadata
    logger.info("Loading training and validation metadata.")
    train_df, valid_df, class_df = load_metadata(config)
    config = add_metadata_info_to_config(config, train_df, valid_df)

    # load model and create optimizer and lr scheduler
    logger.info("Creating model, optimizer, and scheduler.")
    model, model_mean, model_std = load_model(config)

    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    # create dataloaders
    logger.info("Creating DataLoaders.")
    trainloader, validloader, _, _ = get_dataloaders(
        train_df,
        valid_df,
        augmentations=config["augmentations"],
        image_size=config["image_size"],
        model_mean=model_mean,
        model_std=model_std,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
    )

    # create loss function
    class_counts = train_df["class_id"].value_counts().sort_index().values
    criterion = CombinedLoss(class_counts, class_df)

    ## init wandb
    # if wandb_entity is not None and wandb_project is not None:
    #     if resume_exp_name is None:
    #         init_wandb(
    #             config, config["run_name"], entity=wandb_entity, project=wandb_project
    #         )
    #     else:
    #         if "wandb_run_id" not in config:
    #             raise ValueError("Config is missing 'wandb_run_id' field.")
    #         resume_wandb(
    #             run_id=config["wandb_run_id"],
    #             entity=wandb_entity,
    #             project=wandb_project,
    #         )
    #
    # # save config to json in experiment path
    # if resume_exp_name is None:
    #     save_config(config)

    # train model
    logger.info("Training the model.")
    train(
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config["epochs"],
        accumulation_steps=config.get("accumulation_steps", 1),
        clip_grad=config.get("clip_grad"),
        device=device,
        seed=config.get("random_seed", 777),
        path=config["exp_path"],
        resume=resume_exp_name is not None,
        mixup=config.get("mixup"),
        cutmix=config.get("cutmix"),
        mixup_prob=config.get("mixup_prob"),
        apply_ema=config.get("apply_ema"),
        ema_start_epoch=config.get("ema_start_epoch", 0),
        ema_decay=config.get("ema_decay", 0.9999),
    )

    # finish wandb run
    run_id = finish_wandb()
    if run_id is not None:
        logger.info("Setting the best scores in the W&B run summary.")
        set_best_scores_in_summary(
            run_or_path=f"{wandb_entity}/{wandb_project}/{run_id}",
            primary_score="Val. F1",
            scores=lambda df: [col for col in df if col.startswith("Val.")],
        )


if __name__ == "__main__":
    train_clf()
