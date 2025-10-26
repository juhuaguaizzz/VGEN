#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
from transformers import CLIPTextModelWithProjection, CLIPTokenizer,CLIPVisionModelWithProjection
import argparse
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast, BitsAndBytesConfig
import random

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module


# if is_wandb_available():
#     import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# SD3 DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth weights for {base_model}.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

[Download]({repo_id}/tree/main) them in the Files & versions tab.

## License

Please adhere to the licensing terms as described `[here](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE)`.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "sd3",
        "sd3-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two, class_three):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) #8bit量化 在cpu上运行

    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, torch_dtype = torch.float32
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant,torch_dtype = torch.float32
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, quantization_config=quantization_config,variant=args.variant, torch_dtype = torch.float32
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="D:\\python_file\\Quantized-Training-of-SD3\\pre_models",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None, #指定预训练文件的变体
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None, #指定包含训练数据的数据集名称。
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None, #指定数据集的配置名称。如果数据集有多个配置，您可以使用此参数选择特定的配置；如果只有一个配置，可以将其设置为 None。
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None, #指定包含训练数据的文件夹路径。该文件夹应包含用于训练的实例数据。
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None, #下载的模型和数据集将被下载到该目录。
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image", #
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77, #文本最大长度
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None, #验证集提示词
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4, #验证时,一次生成多少图片
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=( #每多少轮运行一次dreambooth的验证
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=6,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=3,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-04, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")




    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    #dataset不能为空
    # if args.dataset_name is None and args.instance_data_dir is None:
    #     raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")
    # #dataset_name和instance_data_dir只能有一个
    # if args.dataset_name is not None and args.instance_data_dir is not None:
    #     raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args

#dreambooth数据集
class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if args.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if args.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = args.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]

            if args.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if args.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][args.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        self.pixel_values = []
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            image = train_transforms(image)
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # custom prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example

#处理逻辑
def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}
    return batch


#提示词数据集
class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

#使用t5编码
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

#使用clip编码
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

#对文本进行编码
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    #返回一个编码结果 和一个 文本池化编码结果
    return prompt_embeds, pooled_prompt_embeds



def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )


    #这是输出文件夹  'sd3-dreambooth/logs'
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir) #配置输出文件夹 和 日志文件夹
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) #分布式训练参数
    #多卡训练配置
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, #设置梯度累计
        mixed_precision=args.mixed_precision, #设计混合精度
        log_with=args.report_to, #设置日志
        project_config=accelerator_project_config, #加速项目配置
        kwargs_handlers=[kwargs], #其他

    )

    device=accelerator.device

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig( #日志记录的配置
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False) #记录了训练状态的所有细节,并且每个进程都会记录日志

    if accelerator.is_local_main_process: #检查当前是否为主进程 然后是一些设置
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None: #如果种子被设定了 那就设置一下种子
        set_seed(args.seed)



    # Handle the repository creation
    if accelerator.is_main_process: #如果当前进程是主进程
        if args.output_dir is not None: #如果没有输出文件夹 就创建输出文件夹
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub: #如果需要推送就推送到huggingface
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers 加载CLIPG
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        torch_dtype = torch.float32,
    )#加载CLIPL
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        torch_dtype = torch.float32,
    )#加载T5xxl
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
        torch_dtype = torch.float32,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    ) #这个地方报错了
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float32
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant,    torch_dtype=torch.float32,
    )
    # 关闭梯度
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    attn_procs = {}  # 用于存储新的架构
    # 这里存储了权重
    trans1 = transformer.state_dict()

    from image_process import ImageProjModel
    from architect_modify import sd3_adapter_attn_processor

    # 创建一个视觉处理器
    image_proj_model = ImageProjModel(  # 独立创建ip-adapter类
        dtype=torch.float32
    )

    image_proj_model=image_proj_model.to("cuda")

    # 替换模块 替换权重
    for name in transformer.attn_processors.keys():

        attn_procs[name] = sd3_adapter_attn_processor()

    transformer.set_attn_processor(attn_procs)



    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float16
    if accelerator.mixed_precision == "fp16": #切换精度
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    #将所有模型放置到cuda上
    vae.to("cuda", dtype=weight_dtype)
    transformer.to("cuda", dtype=weight_dtype)
    image_proj_model.to("cuda",dtype=weight_dtype)
    text_encoder_one.to("cuda", dtype=weight_dtype)
    text_encoder_two.to("cuda", dtype=weight_dtype)
    #clip vision

    clip_vit_h_14_pth="D:\\python_file\\Quantized-Training-of-SD3\\Clip-H"

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_vit_h_14_pth,
                                                                  torch_dtype=weight_dtype).to(accelerator.device)

    text_tokenizer = CLIPTokenizer.from_pretrained(clip_vit_h_14_pth,torch_dtype=weight_dtype)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_vit_h_14_pth,torch_dtype=weight_dtype).cuda()

    # Freeze image_encoder parameters
    for param in image_encoder.parameters():
        param.requires_grad = False

    # Freeze text_encoder parameters
    for param in text_encoder.parameters():
        param.requires_grad = False

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # 保存模型和加载模型的钩子
    def save_model_hook(models, weights, output_dir):#传入模型 权重 输出文件夹
        if accelerator.is_main_process: #在主进程中进行
            transformer_lora_layers_to_save = None

            for model in models: #遍历每一个模型
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model) #如果满足条件就保存lora层
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop() #从weights中移除相应的权重

            StableDiffusion3Pipeline.save_lora_weights( #保存权重并输出
                output_dir, transformer_lora_layers=transformer_lora_layers_to_save
            )

    #加载模型权重
    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)
    #注册钩子
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs, 适用于40 30系显卡 提高训练速度
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if True and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr: #不走这 学习率缩放
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "bf16" or args.mixed_precision == "fp16": #确保训练参数是fp32
        models = [transformer,image_proj_model]
        # models = [transformer]
        # only upcast trainable parameters into fp32
        cast_training_params(models, dtype=torch.float32)




    #通过filter过滤掉不需要梯度的参数 然后作为一个列表返回
    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters 参数+学习率信息组合成一个字典
    transformer_parameters_with_lr = {"params": transformer_parameters, "lr": args.learning_rate}

    # Optimizer creation 检查是prodigy还是adamw 有两种优化器可以选择 如果都不是则强制选择adamw
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"
    #如果使用8bit的adam但是优化器不是adam则发出警告
    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    #如果优化器是adamw
    if True:

        optimizer_class = torch.optim.AdamW #使用adamw作为优化器

        optimizer = optimizer_class( #正常设置
            transformer_parameters+list(image_proj_model.parameters()),
            lr=0.0001,
            betas=(args.adam_beta1, args.adam_beta2), #0.9 0.999
            weight_decay=args.adam_weight_decay, #0.0001
            eps=args.adam_epsilon, #1e-08
        )

        # optimizer = optimizer_class(  # 正常设置
        #     transformer_parameters ,
        #     lr=0.0001,
        #     betas=(args.adam_beta1, args.adam_beta2),  # 0.9 0.999
        #     weight_decay=args.adam_weight_decay,  # 0.0001
        #     eps=args.adam_epsilon,  # 1e-08
        # )


    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    #计算文本嵌入
    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds


    if args.with_prior_preservation:
        class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(
            args.class_prompt, text_encoders, tokenizers
        )



    from torchvision import datasets

    #添加代码

    def resize_with_gaussian_padding(image, target_size=224):
        width, height = image.size

        # **等比例缩放**
        if width > height:
            new_height = target_size
            new_width = int((width / height) * target_size)
        else:
            new_width = target_size
            new_height = int((height / width) * target_size)

        image = image.resize((new_width, new_height), Image.BILINEAR)

        # **计算填充大小**
        pad_left = (target_size - new_width) // 2
        pad_right = target_size - new_width - pad_left
        pad_top = (target_size - new_height) // 2
        pad_bottom = target_size - new_height - pad_top

        # **转换为 Tensor**
        image_tensor = transforms.ToTensor()(image)

        # **生成高斯噪声填充**
        noise = torch.randn(3, target_size, target_size) * 0.02  # 控制噪声强度
        padded_image = F.pad(image_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        padded_image += noise

        return padded_image.clamp(0, 1)  # **确保值在 0-1 之间**

    import cv2
    # 定义数据集路径和转换
    transform = transforms.Compose([
        transforms.Resize((224,224)),  # 随机裁剪不同大小的区域
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    transform_resize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    def add_gaussian_noise(image, mean=0.0):
        """
        向图像添加高斯噪声，并根据指定的概率分布随机选择噪声的标准差。
        :param image: 输入的 PIL 图像
        :param mean: 噪声的平均值
        :return: 添加噪声后的 PIL 图像
        """
        # 定义噪声标准差的概率分布
        prob_intervals = [
            (2.0, 5.0, 0.7),  # 70% 轻度扰动
            (5.0, 10.0, 0.2),  # 20% 中度扰动
            (10.0, 15.0, 0.1)  # 10% 扩展扰动
        ]

        # 从概率分布中随机选择一个区间
        selected_range = random.choices(prob_intervals, [interval[2] for interval in prob_intervals])[0]

        # 随机选择区间内的一个值
        noise_std = random.uniform(selected_range[0], selected_range[1])

        # 将图像转换为 numpy 数组
        image_array = np.array(image).astype(np.float32)

        # 生成与图像相同大小的高斯噪声
        noise = np.random.normal(mean, noise_std, image_array.shape).astype(np.float32)

        # 将噪声加到图像上
        noisy_image = image_array + noise

        # 将值限制在 [0, 255] 之间
        noisy_image = np.clip(noisy_image, 0, 255)

        # 转换回 PIL 图像
        noisy_image = Image.fromarray(np.uint8(noisy_image))
        return noisy_image

    def resize_and_crop(image, target_size=768, noise_std=0.01):
        # 获取原始图像尺寸
        width, height = image.size

        # 计算等比缩放的比例，使短边缩放到512
        if width < height:
            scale = target_size / width
        else:
            scale = target_size / height

        new_width = int(width * scale)
        new_height = int(height * scale)

        # 使用新的尺寸缩放图像
        image_resized = image.resize((new_width, new_height), Image.BILINEAR)

        # 进行中心裁剪，使图像大小为512x512
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = (new_width + target_size) // 2
        bottom = (new_height + target_size) // 2

        image_cropped = image_resized.crop((left, top, right, bottom))

        image_cropped_noise = add_gaussian_noise(image_cropped, mean=0.0,)

        return image_cropped_noise,image_cropped

    # def collate_fn(batch):
    #     tensor_pics = None
    #     tuple_prompts = ()
    #
    #     for tuple_data in batch:
    #         # 获取图像并应用转换
    #         pic = tuple_data[0]
    #
    #         # 将图像最大边缩放到512，保持纵横比，并确保长宽可以被16整除
    #         pic_resized_noise,pic_resized = resize_and_crop(pic, target_size=512)
    #
    #         tensor_pic = transform(pic_resized)  # 这里是用transform转换PIL图像为Tensor
    #         pic_resized = transform_resize(pic_resized_noise)
    #         if tensor_pics is None:
    #             tensor_pics = [pic_resized.unsqueeze(0), tensor_pic.unsqueeze(0)]  # 添加维度，变为 [1, C, H, W]
    #         else:
    #             tensor_pics = [torch.cat((pic_resized.unsqueeze(0),tensor_pics[0]),dim=0), torch.cat((tensor_pic.unsqueeze(0),tensor_pics[1]),dim=0)]
    #
    #         # 随机选择一个提示词
    #         prompts_len = len(tuple_data[1])
    #         index = random.randint(0, prompts_len - 1)
    #         tuple_prompts = tuple_prompts + (tuple_data[1][index],)
    #
    #     return tensor_pics, tuple_prompts

    def collate_fn(batch):
        tensor_pics = None
        tuple_prompts = ()

        for tuple_data in batch:
            # 获取图像并应用转换
            pic = tuple_data[0]

            # 将图像最大边缩放到512，保持纵横比，并确保长宽可以被16整除
            pic_resized_noise, pic_resized = resize_and_crop(pic, target_size=512)

            tensor_pic = transform(pic_resized)  # 这里是用transform转换PIL图像为Tensor
            pic_resized = transform_resize(pic_resized_noise)
            if tensor_pics is None:
                tensor_pics = [pic_resized.unsqueeze(0), tensor_pic.unsqueeze(0)]  # 添加维度，变为 [1, C, H, W]
            else:
                tensor_pics = [torch.cat((pic_resized.unsqueeze(0), tensor_pics[0]), dim=0),
                               torch.cat((tensor_pic.unsqueeze(0), tensor_pics[1]), dim=0)]

            # 随机选择一个提示词
            # prompts_len = len(tuple_data[1])
            # index = random.randint(0, prompts_len - 1)
            tuple_prompts = tuple_prompts + (tuple_data[1],)

        return tensor_pics, tuple_prompts

    # 加载COCO数据集
    train_dataset = datasets.CocoCaptions(root="F:\\coco_dataset\\train2017",
                                          annFile="F:\\coco_dataset\\annotations\\captions_train2017.json",
                                          )

    # 创建数据加载器
    batch_size = 1

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

    # from dataload import CustomDataset

    # folder_path = '/data/datasets/cc3m_unfold'
    # train_dataset = CustomDataset(folder_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_loader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) #计算多少个batch才进行更新
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "sd3-adapter"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    #从之前一个checkpoint加载继续训练
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        # 获取transformer的权重文件
        state_dict = transformer.state_dict()  # 将字典命名为state_dict，避免与Python关键字dict冲突
        # print(state_dict)
        # 加载训练权重
        trained_dict = torch.load(resume_path,weights_only=True)
        # 获取图像映射权重
        image_proj_dict = trained_dict['image_proj_model_dict']
        other_weights = trained_dict['weights_to_save']

        # 加载图像映射权重
        image_proj_model.load_state_dict(image_proj_dict)

        # 创建一个新的字典副本，避免在迭代过程中修改原始字典
        updated_state_dict = state_dict.copy()

        for name in state_dict.keys():
            if "sd3_to_" in name:
                name_temp = name.split("module.")[1]
                weights = other_weights[name]
                updated_state_dict[name] = weights  # 修改副本而非原字典

        # 加载更新后的权重
        transformer.load_state_dict(updated_state_dict)

        initial_global_step = 0
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    #获取sigmas
    def get_sigmas(timesteps, n_dim=4, dtype=weight_dtype):
        #首先获取所以的sigmas
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        #获取所有的时间步
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        #将生成的随机数时间步转移到cuda上
        timesteps = timesteps.to(accelerator.device)
        #将在预设中的时间步全部提取出来作为一个列表
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        #从sigmas列表中根据对应的时间步提取sigmas
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim: #如果小于n_dim
            sigma = sigma.unsqueeze(-1) #增加维度
        return sigma





    loss_list=[]
#开始训练     args.num_train_epochs
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate): #对模型进行梯度累计

                pics_info = batch[0][1].to(dtype=vae.dtype)  # 获取图像
                pixel_values = batch[0][0].to(dtype=vae.dtype)  # 获取图像

                # print(pixel_values.shape)
                random_number = random.randint(0, 4)
                prompts = batch[1][0][random_number]

                text_inputs = text_tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")

                # text_features = text_encoder(**text_inputs).text_embeds

                text_features=text_encoder(**text_inputs)
                # 得到 text embedding（就是 CLIP 的语义向量）
                #新版
                # text_features = text_features.last_hidden_state  # shape: [B, D]
                #旧版
                text_features=text_features.text_embeds

                modified_prompts = []

                # 遍历 prompts 中的每个元素
                # for prompt in prompts:
                #     # 以 5% 的概率将当前元素置为空字符串
                #     if random.random() < 0.15:
                #         modified_prompts.append("")
                #     else:
                #         modified_prompts.append(prompt)


                # 将修改后的 prompts 更新回原来
                # prompts = tuple(text_features)

                # encode batch prompts when custom prompts are provided for each image -

                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(prompts, text_encoders, tokenizers)

                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(weight_dtype)

                # Convert images to latent space
                model_input = vae.encode(pixel_values)  # 将图像编码到潜在空间
                # 获取图像信息
                model_input = model_input.latent_dist.sample()

                model_input = model_input * vae.config.scaling_factor  # 使用vae缩放系数对图像进行编码
                model_input = model_input.to(dtype=weight_dtype)  # 图像迁移到cuda上


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input) #根据输入图像形状 获取一个高斯噪声
                bsz = model_input.shape[0] #batchsize

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                #为每张图像采样一个随机的时间步
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, #根据weighting_scheme计算每个图像的时间步
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long() #根据u来计算时间步 0-1000
                #根据计算出来的时间步下标 获取真实的timesteps
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching. 根据fm添加噪声
                #根据时间步获取sigmas 噪声幅度
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                #加噪 线性加噪
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input #对图像加噪 修正流就是线性加噪
                #使用clip模型处理
                with torch.no_grad():
                    image_embed = image_encoder(pics_info)

                #VGEN正常
                last_hidden_state = image_embed.last_hidden_state.to(weight_dtype)  # [batch_size,257,1280]
                last_hidden_state = last_hidden_state[:, 1:, :]
                text_features=text_features.to(weight_dtype)
                embed_concat,score_clip=image_proj_model(last_hidden_state,text_features)
                # embed_concat=None


                #VGEN使用CLIP最后一层
                # last_hidden_state = image_embed.image_embeds.to(weight_dtype)  # [batch_size,257,1280]
                # text_features = text_features.to(weight_dtype)
                # embed_concat, score_clip = image_proj_model(last_hidden_state, text_features)



                model_pred = transformer(
                    hidden_states=noisy_model_input, #加噪图像
                    timestep=timesteps, #时间步
                    encoder_hidden_states=[prompt_embeds,embed_concat,score_clip], #文本信息
                    pooled_projections=pooled_prompt_embeds, #池化文本信息
                    return_dict=False,
                )[0]

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                # 这一步相当于 预测出向量场 然后将加噪模型朝着向量场 移动sigmas幅度 这样就得到了原始图像

                model_pred = model_pred * (-sigmas) + noisy_model_input

                # model_pred_back=sigmas * noise + (1.0 - sigmas) * model_pred

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                #计算损失加权
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                # 定义目标为原始图像
                target = model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )


                loss = loss.mean()

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                #在这里添加训练采样代码

            logs = {"loss": loss.detach().item(),"lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process and global_step % 100 == 0:
                loss_list.append(loss.detach().item())
                avg_loss = sum(loss_list) / len(loss_list)
                print("this 100 steps average loss:",avg_loss)

                loss_list.clear()
            else:
                loss_list.append(loss.detach().item())
                # loss_back_list.append(loss_back_mean.item())

            if global_step > 340000 == 0:
                accelerator.wait_for_everyone()

                weights_to_save = {}
                transformer_dict = transformer.state_dict()
                image_proj_model_dict = image_proj_model.state_dict()

                for name in transformer_dict.keys():
                    if 'sd3_to_' in name:
                        weights_to_save[name] = transformer_dict[name]

                combined_dict = {
                    'weights_to_save': weights_to_save,
                    'image_proj_model_dict': image_proj_model_dict
                }

                torch.save(combined_dict, f'./save_model/combined_weights-{global_step}.pth')

                return
            if global_step >= args.max_train_steps:
                break




    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        transformer = transformer.to(torch.float32)
        # transformer_lora_layers = get_peft_model_state_dict(transformer)
        weights_to_save = {}
        transformer_dict=transformer.state_dict()
        image_proj_model_dict=image_proj_model.state_dict()

        for name in transformer_dict.keys():
            if 'sd3_to_' in name:
                weights_to_save[name] = transformer_dict[name]

        combined_dict = {
            'weights_to_save': weights_to_save,
            'image_proj_model_dict': image_proj_model_dict
        }

        torch.save(combined_dict, f'./save_model/combined_weights-{global_step}.pth')





    accelerator.end_training()



import os

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    args = parse_args()
    main(args)
