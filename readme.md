# Object Detection with Yolov3


![cover](https://bitmovin.com/wp-content/uploads/2019/08/Object_detection_Blog_Image_Q3_19.jpg)

Object detection is a computer vision task that involves both localizing one or more objects within an image and classifying each object in the image.

It is a challenging computer vision task that requires both successful object localization in order to locate and draw a bounding box around each object in an image, and object classification to predict the correct class of object that was localized.
Yolo is a faster object detection algorithm in computer vision and first described by Joseph Redmon, Santosh Divvala, Ross Girshick and Ali Farhadi in ['You Only Look Once: Unified, Real-Time Object Detection'](https://arxiv.org/abs/1506.02640)

This notebook implements an object detection based on a pre-trained model - [YOLOv3 Pre-trained Weights (yolov3.weights) (237 MB)](https://pjreddie.com/media/files/yolov3.weights).  The model architecture is called a “DarkNet” and was originally loosely based on the VGG-16 model. 

## Predition
```
python detection.py
>>> Where is your image path?
>>> images/traffic.jpg
```
![result](https://raw.githubusercontent.com/patrick013/Object-Detection---Yolov3/master/samples/result.png)

## Details
For details of this project please check [notebook](https://github.com/patrick013/Object-Detection---Yolov3/blob/master/Object_Detection_Yolo.ipynb)

## Reference
> - Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).
> - Darknet, https://github.com/pjreddie/darknet
> - YOLO3 (Detection, Training, and Evaluation), https://github.com/experiencor/keras-yolo3
