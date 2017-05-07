#distributed Tensorflow

**CUDA_ViSIBLE_DEVICES** 
Masking GPUs
시스템에 있는 gpu 장비를 프로그램에 할당 할때 마스킹을 하는 역할을 수행 

CUDA_VISIBLE_DEVICES=0 : 첫번째 장비만 표시 
CUDA_VISIBLE_DEVICES=1 : 두번째 장비만 표시 
CUDA_VISIBLE_DEVICES=0,1 : 첫번째, 두번째 장비 표시 

CUDA의 deviceQuery 실행 결과에 영향을 미쳐서 사용할 수 있는 장비를 바꿈

0 : Titan-X
1 : gtx 970 
0 번째 장비(Titan-x)만 사용할 경우 
<code>
CUDA_VISIBLE_DEVICES=0 python source.py 
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locally
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
name: GeForce GTX TITAN X
major: 5 minor: 2 memoryClockRate (GHz) 1.076
pciBusID 0000:02:00.0
Total memory: 11.92GiB
Free memory: 11.81GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:02:00.0)
</code>

1 번째 장비(gtx970)만 사용할 경우 
<code>
CUDA_VISIBLE_DEVICES=1 python source.py 
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locally
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
name: GeForce GTX 970
major: 5 minor: 2 memoryClockRate (GHz) 1.2155
pciBusID 0000:05:00.0
Total memory: 3.94GiB
Free memory: 3.88GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 970, pci bus id: 0000:05:00.0)
</code>

AWS 분산 텐서 플로우 
https://gist.github.com/haje01/b655a9f0e4b6389b504d6a4e03dea379

https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/how_tos/distributed/
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py


기존의 코드에서 멀티 gpu 이용하기 

tf.device()명령을 이용

<code>
with tf.device("/gpu:0"):
>	weights_1 = tf.Variable(...)
>  	biases_1 = tf.Variable(...)
with tf.device("/gpu:1"):
>  weights_2 = tf.Variable(...)
>  biases_2 = tf.Variable(...)
with tf.device("/gpu:2"):
>  input, labels = ...
>  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
>  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
>  train_op = 
with tf.Session() as sess:
>  for _ in range(10000):
>>    sess.run(train_op)
</code>

tf.Variable, tf.Const를 선언할때 gpu를 지정가능. 

해당 변수 혹은 상수 처리시 정의된 gpu를 사용. 

local server를 이용한 분산 텐서 플로우 

<code>
Start a TensorFlow server as a single-process "cluster".

>import tensorflow as tf
>c = tf.constant("Hello, distributed TensorFlow!")
>server = tf.train.Server.create_local_server()
>sess = tf.Session(server.target)  # Create a session on the server.
>sess.run(c)
'Hello, distributed TensorFlow!'
</code>
