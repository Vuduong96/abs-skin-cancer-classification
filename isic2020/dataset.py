"""Verbatim from ISIC2020_ABLATION_ABS_PATIENT_SPLIT_STOPALLMAL.py's ISIC2020Dataset."""
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ISIC2020Dataset(Dataset):
    """Dataset class for ISIC 2020.
    Expects metadata with columns: image_name, label (int), class_name (str).
    Images are flat .jpg files named <image_name>.jpg.
    """
    def __init__(self, df: pd.DataFrame, images_dir: str,
                 transform=None):
        self.df         = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        img_id     = str(row["image_name"])
        label      = int(row["label"])

        filename = img_id if img_id.endswith(".jpg") else f"{img_id}.jpg"
        path     = os.path.join(self.images_dir, filename)

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, label, img_id
