
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from numpy import expand_dims
from src.process import *
from src.utili import draw_boxes
from src.utili import encoder_dic
from src.nmsupress import do_nms

class Detector():
    def __init__(self,
                 model_path='./model/yolov3.h5'):
        self.model=tf.keras.models.load_model(model_path)
        # print(self.model.summary())

    def _load_and_preprocess_image(self, image_path):
        image = tf.io.read_file(image_path)
        image_width, image_height = load_img(image_path).size
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMAGE_WIDTH,IMAGE_HEIGHT])
        image /= 255

        return image, image_width, image_height

    def _predict(self, image):
        image_x = expand_dims(image, 0)
        yhat = self.model.predict(image_x)
        return yhat

    def _conver_to_boxes(self,yhat, image_width,image_height):
        boxes = list()
        for i in range(len(yhat)):
            boxes += decode_netout(yhat[i][0], ANCHORS[i], net_h=IMAGE_HEIGHT, net_w=IMAGE_WIDTH)

        for i in range(len(boxes)):
            x_offset, x_scale = (IMAGE_WIDTH - IMAGE_WIDTH) / 2. / IMAGE_HEIGHT, float(IMAGE_WIDTH) / IMAGE_WIDTH
            y_offset, y_scale = (IMAGE_HEIGHT - IMAGE_HEIGHT) / 2. / IMAGE_HEIGHT, float(IMAGE_HEIGHT) / IMAGE_HEIGHT
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_width)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_width)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_height)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_height)

        return boxes

    def do_detect(self,image_path):
        image, image_width, image_height=self._load_and_preprocess_image(image_path)
        yhat=self._predict(image)
        boxes=self._conver_to_boxes(yhat,image_width,image_height)
        dic = encoder_dic(box_filter(boxes))
        valid_data=do_nms(dic,NMS_SCORE)
        draw_boxes(image_path,valid_data)

if __name__=="__main__":
    path = input("Where is your image path? \n")
    D=Detector()
    D.do_detect(image_path=path)