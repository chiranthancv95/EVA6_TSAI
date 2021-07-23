# Session10 - 

## Problem Statement - 
### Assignment B:<br>
Download this  downloadfile. Learn how COCO object detection dataset's schema is. This file has the same schema. You'll need to discover what those number are. <br>
Identify these things for this dataset:<br>
readme data for class distribution (along with the class names) along with a graph <br>
Calculate the Anchor Boxes for k = 3, 4, 5, 6 and draw them.<br>
Share the calculations for both via a notebook uploaded on your GitHub Repo <br>

## COCO Dataset - 

COCO is a large-scale object detection, segmentation, and captioning dataset. COCO has several features:<br>

Object segmentation<br>
Recognition in context<br><br>
Superpixel stuff segmentation<br>
330K images (>200K labeled)<br>
1.5 million object instances<br>
80 object categories<br>
91 stuff categories<br>
5 captions per image<br>
250,000 people with keypoints<br>

Classes contained in the Dataset inlcude - <br>    

'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird',<br>
'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports<br> ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine <br>glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot <br>dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell <br>phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'<br>



## Class Distribution for COCO Dataset - 



![class_distribution](./images/dataset_distribution.png)

### Some additional info on the Dataset - 

RangeIndex: 10105 entries, 0 to 10104
Data columns (total 8 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   id           10105 non-null  int64  
 1   height       10105 non-null  float64
 2   width        10105 non-null  float64
 3   x            10105 non-null  float64
 4   y            10105 non-null  float64
 5   bbox_width   10105 non-null  float64
 6   bbox_height  10105 non-null  float64
 7   class        10105 non-null  object 
dtypes: float64(6), int64(1), object(1)
memory usage: 631.7+ KB

## Finding k-means for n_clusters = 3

![3_clusters](./images/3_clusters.png)


## Finding k-means for n_clusters = 4

![4_clusters](./images/4_clusters.png)


## Finding k-means for n_clusters = 5

![5_clusters](./images/5_clusters.png)

## Finding k-means for n_clusters = 6

![6_clusters](./images/6_clusters.png)

## Finding ideal n_clusters through elbow method 

![elbow_method](./images/elbow.png)

## BBOXes based on calculations for all the Ks

![anchor_boxes](./images/anchor_boxes.png)