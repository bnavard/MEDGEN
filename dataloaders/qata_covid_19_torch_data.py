# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np
import pandas as pd
from scipy import ndimage
from typing import Callable
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
from torchvision import transforms as T
from torchvision.transforms import functional as F
from sentence_transformers import SentenceTransformer


def read_text(filename):
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:  # Gets the index of the row number and traverses it
        count = len(df.Description[i].split())
        if count < 9:
            df.Description[i] = df.Description[i] + " EOF XXX" * (9 - count)
        text[df.Image[i]] = df.Description[i]
    return text  # return dict (key: values)

class TextEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("msmarco-distilbert-base-dot-prod-v3")
    def __call__(self, x):
        return self.model.encode(x)

class QataCovid19Dataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str,
        mean: list,
        std: list,
        transform: Callable = None,
    ):
        self.dataset_path = dataset_path
        # select the split 
        if split == "train":
            self.images = os.path.join(dataset_path, "train", "images")
            self.labels = os.path.join(dataset_path, "train", "labels")
            self.raw_text_path = os.path.join(
                dataset_path,
                "train",
                "Train_text.xlsx",
            )
        elif split == "validation":
            self.images = os.path.join(dataset_path, "validation", "images")
            self.labels = os.path.join(dataset_path, "validation", "labels")
            self.raw_text_path = os.path.join(
                dataset_path,
                "validation",
                "Val_text.xlsx",
            )
        else:
            raise ValueError(
                "Specified split is incorrect. Please choose from ['train', 'validation']"
            )
        # get the list of images and masks
        self.images_list = os.listdir(self.images)
        self.mask_list = os.listdir(self.labels)
        assert self.images_list.__len__() == self.mask_list.__len__(), "images and mask count do not match up!"
        
        # text embedding
        self.raw_text = read_text(self.raw_text_path)
        self.text_embedder = TextEmbedder()

        # image and mask transform
        self.transform = transform
        self.normalize = T.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.raw_text) 

    def __getitem__(self, idx):
        mask_filename = self.mask_list[idx]
        image_filename = mask_filename.replace("mask_", "")  
        # read image
        image = read_image(os.path.join(self.images, image_filename), ImageReadMode.RGB)
        
        # embed the text 
        text = self.raw_text[mask_filename]
        text_embedding = self.text_embedder(text)

        # create sample dataset
        sample = {
                "image": image.float(),
                "text_embedding": torch.from_numpy(text_embedding),
        }
        # apply transform to image
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        # And normalize it at the end
        sample["image"] = self.normalize(sample["image"])

        return sample


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # init class
    dataset = QataCovid19Dataset(
        dataset_path="../dataset/QaTa-Covid19/",
        split="train",
    )
    # draw sample
    idx = random.randint(0, dataset.__len__() - 1)
    sample = dataset[idx]
    # reshape
    image = sample["image"].permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    label = sample["label"].permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)

    # Display the first image
    axs[0].imshow(image, cmap="bone")
    axs[0].axis("off")  # Hide the axis
    # axs[0].set_title(f"{image_filename}")
    # Display the second image
    axs[1].imshow(label, cmap="bone")
    axs[1].axis("off")  # Hide the axis
    # axs[1].set_title(f"mask_{image_filename}")
    # Adjust the layout
    plt.tight_layout()
    # Show the plot
    plt.show()
