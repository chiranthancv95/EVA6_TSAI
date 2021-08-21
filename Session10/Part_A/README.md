## Problem Statemnt - 

### Assignment A:
Download this  TINY IMAGENET dataset. <br>
Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. <br>
Submit Results. Of course, you are using your own package for everything. You can look at  this  for reference.<br>


### Model Summary - 

		----------------------------------------------------------------
		Layer (type)               Output Shape         Param #
	================================================================
		    Conv2d-1           [-1, 64, 64, 64]           1,728
	       BatchNorm2d-2           [-1, 64, 64, 64]             128
		    Conv2d-3           [-1, 64, 64, 64]          36,864
	       BatchNorm2d-4           [-1, 64, 64, 64]             128
		    Conv2d-5           [-1, 64, 64, 64]          36,864
	       BatchNorm2d-6           [-1, 64, 64, 64]             128
		BasicBlock-7           [-1, 64, 64, 64]               0
		    Conv2d-8           [-1, 64, 64, 64]          36,864
	       BatchNorm2d-9           [-1, 64, 64, 64]             128
		   Conv2d-10           [-1, 64, 64, 64]          36,864
	      BatchNorm2d-11           [-1, 64, 64, 64]             128
	       BasicBlock-12           [-1, 64, 64, 64]               0
		   Conv2d-13          [-1, 128, 32, 32]          73,728
	      BatchNorm2d-14          [-1, 128, 32, 32]             256
		   Conv2d-15          [-1, 128, 32, 32]         147,456
	      BatchNorm2d-16          [-1, 128, 32, 32]             256
		   Conv2d-17          [-1, 128, 32, 32]           8,192
	      BatchNorm2d-18          [-1, 128, 32, 32]             256
	       BasicBlock-19          [-1, 128, 32, 32]               0
		   Conv2d-20          [-1, 128, 32, 32]         147,456
	      BatchNorm2d-21          [-1, 128, 32, 32]             256
		   Conv2d-22          [-1, 128, 32, 32]         147,456
	      BatchNorm2d-23          [-1, 128, 32, 32]             256
	       BasicBlock-24          [-1, 128, 32, 32]               0
		   Conv2d-25          [-1, 256, 16, 16]         294,912
	      BatchNorm2d-26          [-1, 256, 16, 16]             512
		   Conv2d-27          [-1, 256, 16, 16]         589,824
	      BatchNorm2d-28          [-1, 256, 16, 16]             512
		   Conv2d-29          [-1, 256, 16, 16]          32,768
	      BatchNorm2d-30          [-1, 256, 16, 16]             512
	       BasicBlock-31          [-1, 256, 16, 16]               0
		   Conv2d-32          [-1, 256, 16, 16]         589,824
	      BatchNorm2d-33          [-1, 256, 16, 16]             512
		   Conv2d-34          [-1, 256, 16, 16]         589,824
	      BatchNorm2d-35          [-1, 256, 16, 16]             512
	       BasicBlock-36          [-1, 256, 16, 16]               0
		   Conv2d-37            [-1, 512, 8, 8]       1,179,648
	      BatchNorm2d-38            [-1, 512, 8, 8]           1,024
		   Conv2d-39            [-1, 512, 8, 8]       2,359,296
	      BatchNorm2d-40            [-1, 512, 8, 8]           1,024
		   Conv2d-41            [-1, 512, 8, 8]         131,072
	      BatchNorm2d-42            [-1, 512, 8, 8]           1,024
	       BasicBlock-43            [-1, 512, 8, 8]               0
		   Conv2d-44            [-1, 512, 8, 8]       2,359,296
	      BatchNorm2d-45            [-1, 512, 8, 8]           1,024
		   Conv2d-46            [-1, 512, 8, 8]       2,359,296
	      BatchNorm2d-47            [-1, 512, 8, 8]           1,024
	       BasicBlock-48            [-1, 512, 8, 8]               0
		   Linear-49                  [-1, 200]         102,600
	================================================================
	Total params: 11,271,432
	Trainable params: 11,271,432
	Non-trainable params: 0
	----------------------------------------------------------------
	Input size (MB): 0.05
	Forward/backward pass size (MB): 45.00
	Params size (MB): 43.00
	Estimated Total Size (MB): 88.05
	----------------------------------------------------------------


### Sample Training Data - 

![dataset_sample](https://github.com/chiranthancv95/EVA6_TSAI/blob/main/Session10/Part_A/dataset_sample.png)

### Training logs - 

EPOCH: 37 (LR: 0.002441)
Loss=6.61276 Batch_id=390 Accuracy=6.06: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 38 (LR: 0.002441)
Loss=6.06113 Batch_id=390 Accuracy=14.90: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 39 (LR: 0.003720)
Loss=5.77491 Batch_id=390 Accuracy=21.72: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 40 (LR: 0.005712)
Loss=5.20008 Batch_id=390 Accuracy=27.36: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 41 (LR: 0.008222)
Loss=4.96559 Batch_id=390 Accuracy=32.52: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 42 (LR: 0.011004)
Loss=4.53619 Batch_id=390 Accuracy=36.43: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 43 (LR: 0.013785)
Loss=4.64977 Batch_id=390 Accuracy=40.08: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 44 (LR: 0.016294)
Loss=4.52307 Batch_id=390 Accuracy=42.96: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 45 (LR: 0.018285)
Loss=4.66670 Batch_id=390 Accuracy=46.08: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 46 (LR: 0.019562)
Loss=3.94272 Batch_id=390 Accuracy=48.33: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 47 (LR: 0.020000)
Loss=4.22332 Batch_id=390 Accuracy=50.77: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 48 (LR: 0.019969)
Loss=3.61962 Batch_id=390 Accuracy=52.71: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 49 (LR: 0.019878)
Loss=3.92931 Batch_id=390 Accuracy=54.41: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]

EPOCH: 50 (LR: 0.019726)
Loss=3.84813 Batch_id=390 Accuracy=56.12: 100%|██████████| 391/391 [03:25<00:00,  1.90it/s]


