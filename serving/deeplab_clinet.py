# -*- coding:utf-8 -*-
###
# File: deeplab_clinet.py
# Created Date: 2019-09-20
# Author: jingxiaofei
# Contact: <jingxiaofei@kkworld.com>
# 
# Last Modified: Friday September 20th 2019 4:56:20 pm
# 
# Copyright (c) 2019 KKWorld
# It is never too late to be what you might have been.
# -----
# HISTORY:
# Date      	 By	Comments
# ----------	---	----------------------------------------------------------
###

from __future__ import print_function
from PIL import Image
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import requests
import numpy as np
from StringIO import StringIO
import argparse
import cv2 
from PIL import Image

parser = argparse.ArgumentParser(description='incetion grpc client flags.')
parser.add_argument('--host', default='0.0.0.0', help='inception serving host')
parser.add_argument('--port', default='8500', help='inception serving port')
parser.add_argument('--image', default='/Users/jingxiaofei/Desktop/Images/girl.jpg', help='path to JPEG image file')
FLAGS = parser.parse_args()

def main():  

    # create request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'deeplab'
    request.model_spec.signature_name = 'predict_images'

    # read image into numpy array
    image = np.array(Image.open(FLAGS.image))
    height, width = image.shape[0:2]

    # fill in the request object with the necessary data
    request.inputs['images'].CopyFrom(
    tf.contrib.util.make_tensor_proto(image.astype(dtype=np.float32), shape=[1, height, width, 3]))
    request.inputs['height'].CopyFrom(tf.contrib.util.make_tensor_proto(height, shape=[1]))
    request.inputs['width'].CopyFrom(tf.contrib.util.make_tensor_proto(width, shape=[1]))

    # create prediction service client stub
    channel = implementations.insecure_channel(FLAGS.host, int(FLAGS.port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    result_future = stub.Predict(request, 30.0)

   # get the results
    output = np.array(result_future.outputs['segmentation_map'].int64_val)
    height = result_future.outputs['segmentation_map'].tensor_shape.dim[1].size
    width = result_future.outputs['segmentation_map'].tensor_shape.dim[2].size

    image_mask = np.reshape(output, (height, width)).astype(np.uint8)
    colors = np.unique(image_mask)
    for color in colors:
        if color > 0:
            image_mask[image_mask==color] = 255
    cv2.imwrite('b.png', image_mask)
    
if __name__ == '__main__':
    main()