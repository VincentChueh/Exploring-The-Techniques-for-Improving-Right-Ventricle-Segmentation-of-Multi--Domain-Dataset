import albumentations as A


#custom data input
train_ag=A.Compose([
    #A.HorizontalFlip(p=0.5),
    #A.GaussianBlur(blur_limit=(3,17), p=0.6),
    #gt.ElasticTransform(alpha=8, sigma=60, alpha_affine=50, interpolation=0, value=0, mask_value=None, always_apply=False, approximate=False, same_dxdy=False, p=0.6),
    A.Normalize(mean=(0),std=(1)),
])
val_ag=A.Compose([
    #A.Resize(height=224,width=224),
    A.Normalize(mean=(0),std=(1))
])