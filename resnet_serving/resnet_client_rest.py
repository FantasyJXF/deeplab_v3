#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: resnet_client.py
# Created Date: 2019-09-19
# Author: jingxiaofei
# Contact: <jingxiaofei@kkworld.com>
# 
# Last Modified: Friday September 20th 2019 5:16:18 pm
# 
# Copyright (c) 2019 KKWorld
# It is never too late to be what you might have been.
# -----
# HISTORY:
# Date      	 By	Comments
# ----------	---	----------------------------------------------------------
###

"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from:

https://github.com/tensorflow/models/tree/master/official/resnet#pre-trained-model

The SavedModel must be one that can take JPEG images as inputs.

Typical usage example:

    resnet_client.py
"""

from __future__ import print_function

import base64
import requests
from imagenet_classes import imagenet_classes

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/resnet:predict'

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'http://img.ewebweb.com/uploads/20190403/14/1554274318-AZHxeCFcIK.jpeg'


def main():
  # Download the image
  dl_request = requests.get(IMAGE_URL, stream=True)
  dl_request.raise_for_status()

  # Compose a JSON Predict request (send JPEG image in base64).
  # According to https://www.tensorflow.org/tfx/serving/api_rest
  # JSON uses UTF-8 encoding. If you have input feature or tensor values that need to be binary 
  # (like image bytes), you must Base64 encode the data and encapsulate it in a JSON object 
  # having b64 as the key as follows:
  # { "b64": <base64 encoded string> }
  jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
  predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes

  # Send few requests to warm-up the model.
  for _ in range(3):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()

  # Send few actual requests and report average latency.
  total_time = 0
  num_requests = 10
  for _ in range(num_requests):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds()
    prediction = response.json()['predictions'][0]

    print('Prediction class: ({}: {}), avg latency: {} ms'.format(
      prediction['classes']-1, imagenet_classes[prediction['classes']-1], (total_time*1000)/num_requests))


if __name__ == '__main__':
  main()