import albumentations as A
import albumentations.augmentations.geometric.transforms as gt
train_ag=A.Compose([
    #A.HorizontalFlip(p=0.5),
    #A.GaussianBlur(blur_limit=(3,17), p=0.6),
    A.GaussNoise(var_limit=(0, 1), mean=0, per_channel=True, always_apply=False, p=0.1),
    gt.OpticalDistortion(distort_limit=0.9, shift_limit=0.2, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.2),
    #A.augmentations.geometric.transforms.ElasticTransform (alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=0.5)
    #A.Normalize(mean=(0),std=(1)),
])
val_ag=A.Compose([
    #A.Resize(height=224,width=224),
    #A.Normalize(mean=(0),std=(1))
])