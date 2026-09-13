"""Verbatim from every reported ABS run's transform builders --
IMG_SIZE=300, this exact augmentation recipe, is what produced every
ABS number in the paper."""
import torchvision.transforms as T


def build_aggressive_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0),
                            ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.05)
        ], p=0.7),
        T.RandomApply([
            T.RandomAffine(degrees=45, translate=(0.1, 0.1),
                           scale=(0.8, 1.2), shear=10)
        ], p=0.6),
        T.RandomApply([
            T.RandomPerspective(distortion_scale=0.3)
        ], p=0.3),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def build_val_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
