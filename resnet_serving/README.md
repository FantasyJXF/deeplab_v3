# TF Serving

## Server

```
$ mkdir /tmp/resnet
$ curl -s https://storage.googleapis.com/download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | tar --strip-components=2 -C /tmp/resnet -xvz
$ docker pull tensorflow/serving
$ docker run -p 8501:8501 -p 8500:8500 --name tfserving_resnet \
--mount type=bind,source=/Users/jingxiaofei/Documents/Learning/TensorFlow/deeplab_v3/resnet_serving/resnet,target=/models/resnet \
-e MODEL_NAME=resnet -t tensorflow/serving &
```

Breaking down the command line arguments, we are:
* -p 8501:8501 -p 8500:8500 : Publishing the container’s port 8501 (where TF Serving responds to REST API requests) to the host’s port 8501. Use both gRPC and RESTful
* --name tfserving_resnet : Giving the container we are creating the name “tfserving_resnet” so we can refer to it later
* --mount type=bind,source=<path to your model>,target=/models/resnet : Mounting the host’s local directory (<path to your model> 本机路径，在variables上2层) on the container (/models/resnet) so TF Serving can read the model from inside the container. 将本机模型文件与docker里的容器进行绑定，相当于软连接还是移动？
* -e MODEL_NAME=resnet : Telling TensorFlow Serving to load the model named “resnet”
* -t tensorflow/serving : Running a Docker container based on the serving image “tensorflow/serving”

### 查看模型的输入输出
很多时候我们需要查看模型的输出和输出参数的具体形式，TensorFlow提供了一个`saved_model_cli`命令来查看模型的输入和输出参数：

> 这里的`dir`需要是具体到版本号的模型路径

```
$ saved_model_cli show --dir /tmp/resnet/1538687457/ --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['predict']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['image_bytes'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['classes'] tensor_info:
        dtype: DT_INT64
        shape: (-1)
        name: ArgMax:0
    outputs['probabilities'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1001)
        name: softmax_tensor:0
  Method name is: tensorflow/serving/predict

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['image_bytes'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: input_tensor:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['classes'] tensor_info:
        dtype: DT_INT64
        shape: (-1)
        name: ArgMax:0
    outputs['probabilities'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1001)
        name: softmax_tensor:0
  Method name is: tensorflow/serving/predict

```

注意到signature_def，inputs的名称，类型和输出，这些参数在接下来的模型预测请求中需要。

## Client
客户端的作用是将图片传到服务端的模型中进而得到预测结果，例如处理完成的图片。

### gRPC接口
参考`resnet_client_grpc.py`

### RESTfull接口
参考`resnet_client_rest.py`

### 

> ImageNet的10000类具体标签参看`imagenet_classes.py`
> TF Serving返回值类型为google.protobuf.pyext._message.RepeatedScalarContainer，一般是以数组的形式返回，取值后用列表索引取值后就可以进行普通计算了

## Performance性能
通过编译优化的TensorFlow Serving 二进制来提高性能
TensorFlows serving有时会有输出如下的日志：
```
Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
```
复制代码TensorFlow Serving已发布Docker镜像旨在尽可能多地使用CPU架构，因此省略了一些优化以最大限度地提高兼容性。如果你没有看到此消息，则你的二进制文件可能已针对你的CPU进行了优化。根据你的模型执行的操作，这些优化可能会对你的服务性能产生重大影响。幸运的是，编译优化的TensorFlow Serving二进制非常简单。官方已经提供了自动化脚本，分以下两部进行：

# 1. 编译开发版本
```
$ docker build -t $USER/tensorflow-serving-devel -f Dockerfile.devel https://github.com/tensorflow/serving.git#:tensorflow_serving/tools/docker

# 2. 生产新的镜像
$ docker build -t $USER/tensorflow-serving --build-arg TF_SERVING_BUILD_IMAGE=$USER/tensorflow-serving-devel https://github.com/tensorflow/serving.git#:tensorflow_serving/tools/docker
```
复制代码之后，使用新编译的$USER/tensorflow-serving重新启动服务即可。
