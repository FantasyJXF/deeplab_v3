from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import requests
import base64
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

    channel = implementations.insecure_channel(FLAGS.host, int(FLAGS.port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Send request
    # See prediction_service.proto for gRPC request/response details.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'deeplab'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['input_bytes'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=[1]))
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    
    
    encoded_response_string = result.outputs['output_bytes'].string_val[0]
    encoded_response_string = bytes(encoded_response_string)
    #response_image = base64.b64decode(encoded_response_string)

    # Save inferred image
    with open("lol11111.png", "wb") as output_file:
        output_file.write(encoded_response_string)

if __name__ == '__main__':
  tf.app.run()