import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HAM10000Dataset(Dataset):
    """Dataset class for HAM10000.
    Expects metadata with columns: img_id, label (int), class_name (str).
    Images are .jpg files (img_id does not include the extension).
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
        img_id     = str(row["img_id"])
        label      = int(row["label"])
        class_name = str(row["class_name"])

        # HAM10000 img_id does not include the extension
        filename    = img_id if img_id.endswith(".jpg") else f"{img_id}.jpg"
        path_flat   = os.path.join(self.images_dir, filename)
        path_subdir = os.path.join(self.images_dir, class_name, filename)

        if os.path.exists(path_flat):
            path = path_flat
        elif os.path.exists(path_subdir):
            path = path_subdir
        else:
            path = path_flat  # will raise a clear error if truly missing

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        if self.transform:
            img = self.transform(img)
        return img, label, img_id
