# TF Serving

### Server

```
$ mkdir /tmp/resnet
$ curl -s https://storage.googleapis.com/download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | tar --strip-components=2 -C /tmp/resnet -xvz
$ docker pull tensorflow/serving
$ docker run -p 8501:8501 --name tfserving_resnet \
--mount type=bind,source=/Users/jingxiaofei/Documents/Learning/TensorFlow/resnet_serving/resnet,target=/models/resnet \
-e MODEL_NAME=resnet -t tensorflow/serving &
```

Breaking down the command line arguments, we are:
* -p 8501:8501 : Publishing the container’s port 8501 (where TF Serving responds to REST API requests) to the host’s port 8501
* --name tfserving_resnet : Giving the container we are creating the name “tfserving_resnet” so we can refer to it later
* --mount type=bind,source=<path to your model>,target=/models/resnet : Mounting the host’s local directory (<path to your model> 本机路径，在variables上2层) on the container (/models/resnet) so TF Serving can read the model from inside the container. 将本机模型文件与docker里的容器进行绑定，相当于软连接还是移动？
* -e MODEL_NAME=resnet : Telling TensorFlow Serving to load the model named “resnet”
* -t tensorflow/serving : Running a Docker container based on the serving image “tensorflow/serving”

### Client
客户端的作用是将图片传到服务端的模型中进而得到预测结果，例如处理完成的图片。
在`resnet_client.py`中

> ImageNet的10000类具体标签