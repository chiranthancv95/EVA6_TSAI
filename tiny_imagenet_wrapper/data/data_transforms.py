from albumentations import (
	Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    PadIfNeeded,
    RGBShift,
    Rotate,
    Cutout
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision.transforms as transforms

def albumentations_transforms_old(p=1.0, is_train=False):
	# Mean and standard deviation of train dataset
	mean = np.array([0.4914, 0.4822, 0.4465])
	std = np.array([0.2023, 0.1994, 0.2010])
	#transforms_list = []
	# Use data aug only for train data
	if is_train:
		train_transforms = [
			A.Normalize(mean=mean, std=std),
			A.PadIfNeeded(min_height=40, min_width=40, border_mode=4, always_apply=True, p=1.0),
			A.RandomCrop (32, 32, always_apply=True, p=1.0),
			A.HorizontalFlip(p=0.5),
			A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, always_apply=False, p=1),
			ToTensorV2()
			]
	data_transforms = Compose(train_transforms, p=p)
	return lambda img: data_transforms(image=np.array(img))["image"]

def albumentations_transforms(p=1.0, is_train=False):
    '''Applies image augmentations to image dataset 
    RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
    
    Returns:
        list of transforms'''
    mean = (0.491, 0.482, 0.446)
    std = (0.2302, 0.2265, 0.2262)
    mean = np.mean(mean)
    train_transforms = [
        A.Normalize(mean=mean, std=std),
        A.PadIfNeeded(min_height=40, min_width=40, border_mode=4, always_apply=True, p=1.0),
        A.RandomCrop (32, 32, always_apply=True, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value = None),
        ToTensorV2()
    ]
    transforms_result = A.Compose(train_transforms)
    return lambda img:transforms_result(image=np.array(img))["image"]

def albumentations_transforms_test():
    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2302, 0.2265, 0.2262]

    test_transform = [
        A.Normalize(mean=mean, std=std),
       ToTensorV2()]
    transforms_result = A.Compose(test_transform)
    return lambda img:transforms_result(image=np.array(img))["image"]

# def torch_transforms(is_train=False):
# 	# Mean and standard deviation of train dataset
# 	mean = (0.4914, 0.4822, 0.4465)
# 	std = (0.2023, 0.1994, 0.2010)
# 	transforms_list = []
# 	# Use data aug only for train data
# 	if is_train:
# 		transforms_list.extend([
# 			transforms.RandomCrop(64, padding=4),
# 			transforms.RandomHorizontalFlip(),
# 		])
# 	transforms_list.extend([
# 		transforms.ToTensor(),
# 		transforms.Normalize(mean, std),
# 	])
# 	if is_train:
# 		transforms_list.extend([
# 			transforms.RandomErasing(0.25)
# 		])
# 	return transforms.Compose(transforms_list)
