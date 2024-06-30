import os
import logging
import torch.nn.functional as F
import torch.nn as nn
from fgvc.datasets import get_dataloaders
from torch.utils.data import DataLoader
from fgvc.losses import SeesawLossWithLogits
from fgvc.core.training import train, predict
from clas_train_cust import ClassificationTrainer
from fgvc.utils.experiment import (
    get_optimizer_and_scheduler,
    load_args,
    load_config,
    load_model,
    load_train_metadata,
    save_config,
)
import timm
from fgvc.utils.utils import set_cuda_device, set_random_seed
from fgvc.utils.wandb import (
    finish_wandb,
    init_wandb,
    resume_wandb,
    set_best_scores_in_summary,
)
from fgvc.core.augmentations import (vit_heavy_transforms)
from typing import Tuple, Union
import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("script")

class CustomImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Union[A.Compose, T.Compose], **kwargs):
        assert "image_path" in df
        assert "class_id" in df
        assert "MIVS" in df
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        image, file_path = self.get_image(idx)
        class_id, mvis = self.get_class_id(idx)
        image = self.apply_transforms(image)
        return image, np.array([class_id, mvis]), file_path

    def get_image(self, idx: int) -> Tuple[Image.Image, str]:
        """Get i-th image and its file path in the dataset."""
        file_path = self.df["image_path"].iloc[idx]
        image_pil = Image.open(file_path).convert("RGB")
        # if len(image_pil.size) < 3:
        #     rgbimg = Image.new("RGB", image_pil.size)
        #     rgbimg.paste(image_pil)
        #     image_pil = rgbimg
        # image_np = np.asarray(image_pil)[:, :, :3]
        return image_pil, file_path

    def get_class_id(self, idx: int) -> int:
        """Get class id of i-th element in the dataset."""
        return self.df["class_id"].iloc[idx], self.df["MIVS"].iloc[idx]

    def apply_transforms(self, image: Image.Image) -> torch.Tensor:
        """Apply augmentation transformations on the image."""
        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=np.asarray(image))["image"]
            else:
                image = self.transform(image)
        return image

class CustomModel(nn.Module):
    def __init__(self, base_model_name, num_classes1, num_classes2):
        super(CustomModel, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True)
        in_features = self.base_model.get_classifier().in_features
        self.base_model.reset_classifier(0)  # Remove the original classification layer

        self.fc1 = nn.Linear(in_features, num_classes1)  # Binary classification output
        self.fc2 = nn.Linear(in_features, num_classes2)  # Categorical classification output

    def forward(self, x):
        x = self.base_model(x)
        out1 = torch.sigmoid(self.fc1(x))  # Binary output
        out2 = self.fc2(x)  # Categorical output
        return out1, out2

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.seesaw_loss = SeesawLossWithLogits(class_counts=1784)
        self.num_labels = 1784

    def forward(self, outputs, targets):
        out1 = torch.flatten(outputs[0])  # Binary output
        target1 = targets[:, 1].type(torch.float32)  # Binary target

        out2 = outputs[1] # Categorical output
        target2 = targets[:, 0] # Categorical target

        loss1 = self.bce_loss(out1, target1)
        loss2 = self.seesaw_loss(out2, target2)
        return loss1 + loss2


def load_metadata(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load metadata of the traning and validation sets."""
    assert "dataset" in config

    BASE_IMAGE_PATH_iNAT = "/media/toofy/kky_plzen4/projects/korpusy_cv/SnakeCLEF2024/train/SnakeCLEF2023-large_size"
    # BASE_IMAGE_PATH_HMP = "/media/toofy/kky_plzen4/train/"
    BASE_IMAGE_PATH_VAL= "/media/toofy/kky_plzen4/projects/korpusy_cv/SnakeCLEF2024/val/SnakeCLEF2023-large_size"

    train_metadata_iNat = pd.read_csv("./metadata/SnakeCLEF2023-Train-iNat-cleared.csv")
    # train_metadata_hmp = pd.read_csv("./metadata/SnakeCLEF2023-TrainMetadata-HM.csv")
    val_metadata = pd.read_csv("./metadata/SnakeCLEF2023-ValMetadata.csv")
    venomous_metadata = pd.read_csv("./metadata/venomous_status_list.csv")

    train_metadata_iNat["image_path"] = train_metadata_iNat.image_path.apply(
        lambda path: os.path.join(BASE_IMAGE_PATH_iNAT, path)
    )

    val_metadata["image_path"] = val_metadata.image_path.apply(
        lambda path: os.path.join(BASE_IMAGE_PATH_VAL, path)
    )

    # train_metadata_hmp["image_path"] = train_metadata_hmp.image_path.apply(
    #     lambda path: os.path.join(BASE_IMAGE_PATH_HMP, path)
    # )

    metadata = pd.concat(
        [train_metadata_iNat, val_metadata]).reset_index(drop=True)  # train_metadata_hmp

    metadata_o = pd.merge(metadata, venomous_metadata, left_on='class_id', right_on='class_id', how='left')

    train_df = metadata_o[metadata_o.subset != "val"]
    valid_df = metadata_o[metadata_o.subset == "val"]

    return train_df, valid_df

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
    train_df, valid_df = load_metadata(config)
    config = add_metadata_info_to_config(config, train_df, valid_df)

    # load model and create optimizer and lr scheduler
    logger.info("Creating model, optimizer, and scheduler.")
    # model, model_mean, model_std = load_model(config)
    model = CustomModel(base_model_name='swinv2_tiny_window16_256.ms_in1k', num_classes1=1, num_classes2=len(train_df.class_id.unique()))
    model_mean = tuple(model.base_model.default_cfg["mean"])
    model_std = tuple(model.base_model.default_cfg["std"])

    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    # create dataloaders
    logger.info("Creating DataLoaders.") #TODO create dataloaders

    transforms_kws = {}
    transforms_fn = vit_heavy_transforms
    train_tfm, val_tfm = transforms_fn(image_size=config["image_size"], mean=model_mean, std=model_std, **transforms_kws)


    trainset = CustomImageDataset(train_df, transform=train_tfm)
    trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=config["workers"])
    valset = CustomImageDataset(valid_df, transform=val_tfm)
    validloader = DataLoader(valset, batch_size=config["batch_size"], shuffle=False, num_workers=config["workers"])

    logger.info("Creating loss function.")
    criterion = CombinedLoss() # custom loss function

    # init wandb
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

    # save config to json in experiment path
    if resume_exp_name is None:
        save_config(config)

    # train model
    logger.info("Training the model.")
    trainer = ClassificationTrainer(
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

    trainer.train(
        num_epochs=config["epochs"],
        seed=config.get("random_seed", 777),
        path=config["exp_path"],
        resume=resume_exp_name is not None,
        **{},
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
