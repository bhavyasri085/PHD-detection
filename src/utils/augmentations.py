import albumentations as A


def get_train_transforms(img_size=(480, 640)):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, border_mode=0, p=0.4),
    ], bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category_ids"],
        min_visibility=0.3,
    ))


def get_val_transforms(img_size=(480, 640)):
    return None