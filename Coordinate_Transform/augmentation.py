import albumentations as A
import albumentations.augmentations.geometric.transforms as gt

train_ag=A.Compose([
    #A.Resize(height=224,width=224),
    #A.RandomCrop(224,224,0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=1.0),
    A.Blur(10, p=1.0),
    gt.ElasticTransform(alpha=1, sigma=50, alpha_affine=25, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=0.5),
    A.Normalize(mean=(0),std=(1)),
])
val_ag=A.Compose([
    #A.Resize(height=224,width=224),
    A.Normalize(mean=(0),std=(1))
])