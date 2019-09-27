# -*- coding:utf-8 -*-
###
# File: deeplab_client_rest.py
# Created Date: 2019-09-20
# Author: jingxiaofei
# Contact: <jingxiaofei@kkworld.com>
# 
# Last Modified: Friday September 27th 2019 10:19:42 am
# 
# Copyright (c) 2019 KKWorld
# It is never too late to be what you might have been.
# -----
# HISTORY:
# Date      	 By	Comments
# ----------	---	----------------------------------------------------------
###

from __future__ import print_function

import os
import base64
import requests
import json
import argparse
import numpy as np
import utils
import cv2 as cv
import ipdb

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in-path", default='https://img.gq.com.tw/_rs/645/userfiles/sm/sm1024_images_A1/40721/2019092061159601.jpg', help="Input Image.")
    parser.add_argument("--out-path", default='lol.jpg', help="Output Image.")
    parser.add_argument("--host", default="0.0.0.0", help="Host address.")
    parser.add_argument("--port", default="9001", help="RESTful API port.")
    parser.add_argument("--model", default="deeplab_v1", help="Serving model.")
    args = parser.parse_args()
    return args

def main(args):
    # The input is URL
    if args.in_path.startswith("http"):
        dl_request = requests.get(args.in_path, stream=True)
        dl_request.raise_for_status()
        input_image = dl_request.content
    # Input image
    elif os.path.exists(args.in_path):
        input_image = open(args.in_path, "rb").read()
    else:
        print("File doesn't exists")
        raise Exception("No such file or url")

    # Compose a JSON Predict request (send JPEG image in base64).
    # According to https://www.tensorflow.org/tfx/serving/api_rest
    # JSON uses UTF-8 encoding. If you have input feature or tensor values that need to be binary 
    # (like image bytes), you must Base64 encode the data and encapsulate it in a JSON object 
    # having b64 as the key as follows:
    # { "b64": <base64 encoded string> }
    jpeg_bytes = base64.b64encode(input_image).decode("utf-8")
    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes

    # # Send few requests to warm-up the model.
    # for _ in range(3):
    #     response = requests.post(SERVER_URL, data=predict_request)
    #     response.raise_for_status()
    SERVER_URL = "http://{}:{}/v1/models/{}:predict".format(args.host, args.port, args.model)
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()

    # Extract text from JSON
    response = json.loads(response.text)

    # Interpret bitstring output
    response_string = response["predictions"][0]["b64"]

    # Decode bitstring
    encoded_response_string = response_string.encode("utf-8")
    image_bitstring = base64.b64decode(encoded_response_string)

    img_np = np.frombuffer(image_bitstring, dtype=np.uint8)
    img_np = cv.imdecode(img_np, flags=1)[:,:,0]
    img_bgr = utils.decode_segmap(img_np)
    cv.imwrite(args.out_path, img_bgr)
    
    #Save inferred image
    # with open(args.out_path, "wb") as output_file:
    #     output_file.write(image_bitstring)

if __name__ == '__main__':
    args = parse_args()
    main(args)