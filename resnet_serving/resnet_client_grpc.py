#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: resnet_client_grpc.py
# Created Date: 2019-09-20
# Author: jingxiaofei
# Contact: <jingxiaofei@kkworld.com>
# 
# Last Modified: Friday September 20th 2019 6:10:25 pm
# 
# Copyright (c) 2019 KKWorld
# It is never too late to be what you might have been.
# -----
# HISTORY:
# Date      	 By	Comments
# ----------	---	----------------------------------------------------------
###

"""Send JPEG image to tensorflow_model_server loaded with ResNet model.

"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import requests
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf.app.flags.DEFINE_string('host', '0.0.0.0', "localhost")
tf.app.flags.DEFINE_string('port', '8500', 'port number')
FLAGS = tf.app.flags.FLAGS


def main(_):
    if FLAGS.image:
        with open(FLAGS.image, 'rb') as f:
            data = f.read()
    else:
        # Download the image since we weren't given one
        dl_request = requests.get(IMAGE_URL, stream=True)
        dl_request.raise_for_status()
        data = dl_request.content

    #channel = grpc.insecure_channel(FLAGS.server)
    channel = implementations.insecure_channel(FLAGS.host, int(FLAGS.port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Send request
    # See prediction_service.proto for gRPC request/response details.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'resnet'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['image_bytes'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=[1]))
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    #print(result)
    print('Prediction class: {}'.format(result.outputs['classes'].int64_val[0] - 1))

if __name__ == '__main__':
  tf.app.run()