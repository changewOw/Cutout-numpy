# Cutout-numpy
# Usage:
    aug = Cutout(2, 100, 50, fill_value_mode='uniform',p=1)
    img_c, mask_c = aug(img=image, semantic_label=mask)
    only support semantic segmentation task
Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    |  https://github.com/albu/albumentations
   
