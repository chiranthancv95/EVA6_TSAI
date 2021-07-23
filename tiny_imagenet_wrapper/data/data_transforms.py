from albumentations import (
	Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    PadIfNeeded,
    RGBShift,
    Rotate
)
from albumentations.pytorch import ToTensor
import numpy as np
import torchvision.transforms as transforms

def albumentations_transforms(p=1.0, is_train=False):
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
			A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=mean, always_apply=False, p=1),
			ToTensor()
			]
	data_transforms = Compose(train_transforms, p=p)
	return lambda img: data_transforms(image=np.array(img))["image"]

def torch_transforms(is_train=False):
	# Mean and standard deviation of train dataset
	mean = (0.4914, 0.4822, 0.4465)
	std = (0.2023, 0.1994, 0.2010)
	transforms_list = []
	# Use data aug only for train data
	if is_train:
		transforms_list.extend([
			transforms.RandomCrop(64, padding=4),
			transforms.RandomHorizontalFlip(),
		])
	transforms_list.extend([
		transforms.ToTensor(),
		transforms.Normalize(mean, std),
	])
	if is_train:
		transforms_list.extend([
			transforms.RandomErasing(0.25)
		])
	return transforms.Compose(transforms_list)
