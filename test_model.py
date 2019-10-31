#coding=utf-8
import numpy as np
import cv2
import pickle
from keras.layers import Dense, Input, BatchNormalization, Conv2D, Flatten, MaxPooling2D, Activation, Reshape, Layer
from keras.models import Model, load_model, model_from_json
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import sys

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z"]


class NormLayer(Layer):

    def __init__(self, **kwargs):
        super(NormLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernal = self.add_weight(name='NormLayer', shape=(
            1, 13), initializer='ones', trainable=True)
        super(NormLayer, self).build(input_shape)

    def call(self, inputs):
        # out = self.kernal * inputs
        out = K.dot(self.kernal, inputs)
        out = K.permute_dimensions(out, (1, 0, 2))
        return out[:, 0, :]
    ''' because this NormLayer do not change the input_shape,
        so the compute_output_shape need not to implement (maybe)
    '''

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 65)

    def get_config(self):
        # config = {}
        base_config = super(NormLayer, self).get_config()
        return dict(list(base_config.items()))
        # return dict(list(base_config.items()) + list(config.items()))


def print_predictLabel(x):
    num = x[0].shape[0]
    sort = np.zeros([num, len(x)])
    for i in range(len(x)):
        temp = x[i]
        sort[:, i] = np.argmax(temp, 1)
    for i in range(num):
        print(np.array([int(sort[i, 0]), int(sort[i, 1] + 31), int(sort[i, 2] + 31), int(sort[i, 3] + 31), int(sort[i, 4] + 31), int(sort[i, 5] + 31), int(sort[i, 6] + 31)]))
        print(chars[int(sort[i, 0])] + chars[int(sort[i, 1] + 31)] + chars[int(sort[i, 2] + 31)] +
              chars[int(sort[i, 3] + 31)] + chars[int(sort[i, 4] + 31)] + chars[int(sort[i, 5] + 31)] +
              chars[int(sort[i, 6] + 31)])
        
def cropImage(image,rect):
        x, y, w, h = computeSafeRegion(image.shape,rect)
        #cv2.imwrite('./chepai.jpg', image[y:y+h,x:x+w])
        return image[y:y+h,x:x+w]
 
def computeSafeRegion(shape,bounding_rect):
        top = bounding_rect[1] # y
        bottom  = bounding_rect[1] + bounding_rect[3] # y +  h
        left = bounding_rect[0] # x
        right =   bounding_rect[0] + bounding_rect[2] # x +  w
        min_top = 0
        max_bottom = shape[0]
        min_left = 0
        max_right = shape[1]
        #print(left,top,right,bottom)
        #print(max_bottom,max_right)
        if top < min_top:
            top = min_top
        if left < min_left:
            left = min_left
        if bottom > max_bottom:
            bottom = max_bottom
        if right > max_right:
            right = max_right
        return [left,top,right-left,bottom-top]  

def detectPlateRough(image_gray,resize_h = 720,en_scale =1.08 ,top_bottom_padding_rate = 0.05):
        if top_bottom_padding_rate>0.2:
            print("error:top_bottom_padding_rate > 0.2:",top_bottom_padding_rate)
            exit(1)
        height = image_gray.shape[0]
        watch_cascade = cv2.CascadeClassifier('./cascade.xml')
        padding = int(height*top_bottom_padding_rate)
        scale = image_gray.shape[1]/float(image_gray.shape[0])
        image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))
        image_color_cropped = image[padding:resize_h-padding,0:image_gray.shape[1]]
        image_gray = cv2.cvtColor(image_color_cropped,cv2.COLOR_RGB2GRAY)
        watches = watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),maxSize=(36*40, 9*40))
        cropped_images = []
        for (x, y, w, h) in watches:
            x -= w * 0.14
            w += w * 0.28
            y -= h * 0.6
            h += h * 1.1;
            #y -= h * 0.15
            #h += h * 0.3
            cropped = cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
            #cropped_images.append([cropped,[x, y+padding, w, h]])
        #return cropped_images
        return cropped

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_nn.py <plate_image>")
        exit()
    img_path = sys.argv[1]
    e2e_model = load_model('./e2e_model.h5', custom_objects={'NormLayer': NormLayer})
    image = cv2.imread(img_path)
    img_detect = detectPlateRough(image,image.shape[0],top_bottom_padding_rate=0.1)
    batchSize=1
    img_data = np.zeros([batchSize, 30, 120, 3])
    img_temp = np.reshape(img_detect, (30, 120, 3))
    img_data[0, :, :, :] = img_temp
    pkl_data_path = '/tmp/test_train_data.pkl'
    output = open(pkl_data_path, 'wb', )
    pickle.dump(img_data, output, protocol = 4)

    output.close()

    data = np.load(pkl_data_path)
    data = data.transpose(0, 2, 1, 3)
    e2e_predict = e2e_model.predict(data[0:1, :, :, :])

    print_predictLabel(e2e_predict)

