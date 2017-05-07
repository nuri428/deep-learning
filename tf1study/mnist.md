# MNIST
### 시그모이드 함수(Sigmoid)
+ 정의 
    * S자 모양의 함수. 0과 1에 무한히 가까워지는 함수로, 인공신경망의 뉴런에서 일어나는 선형적인 가중치 계산을 비선형적으로 변경시켜주어 폭넒은 문제에 적용가능

+ 수식  
    * 값이 가장 작아도 0에 수렴하고, 가장 큰 값도 1에 수렴하는 함수  
       ![sigmoid](http://postfiles3.naver.net/20150612_50/2feelus_14340467064157goJq_PNG/2015-06-12_at_3.21.28_AM.png?type=w2)  

+ 그래프  

![sigmoidGraph](http://postfiles8.naver.net/20150612_71/2feelus_14340466751522xoTj_PNG/2015-06-12_at_3.20.33_AM.png?type=w2)


### 소프트맥스(SoftMax)
+ 정의 :
    * input x가 주어졌을 때 이것이 i일거라고 확신하는 정도(evidence)  
    * n개의 값을 예측할 때 사용
    * n개의 확률의 합은 1
+ 수식    
    - 입력 x가 주어졌을때 i에 대한 근거
    - i는 클래스를 나타내며, j는 입력이미지의 픽셀들의 합계를 내기위한 인덱스,W는 가중치
    - i(0~9까지 숫자)에 대해 784개의 원소(28*28)를 가지는 행렬 Wi를 얻게됨
    - Wi는 784개 픽셀로 이루어진 입력 이미지 원소들과 곱해지고, 끝으로 bi를 더함  

        ![softMax](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-76767ba1433d447c496fb1ae236967e1_l3.png)  
    - 소프트맥스 함수를 사용하여 근거들의 합을 예측하는 확률 y

        ![sigmoid](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-dee7412635e21a5494d11c8110245f7e_l3.png)  
    - 위에서 출력 벡터는 원소의 합이 1인 확률 함수가 되어야 하기 때문에 각 벡터의 원소를 정규화하기 위해 소프트맥스 함수는 입력값을 모두 지수 값으로 바꿔줌
    - 지수함수를 사용하면 가중치를 더 커지게 하는 효과를 얻음
    - 또한, 한 클래스의 근거가 작을 때 이 클래스의 확률도 더 낮아지게 됨
    - 따라서 소프트맥스 함수는 가중치의 합이 1이 되도록 정규화하여 확률분포를 만들어줌

        ![sigmoid](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-3529273538223cea9fc696ee62202649_l3.png)  

![sigmoid](http://solarisailab.com/wp-content/uploads/2016/05/softmax-regression-scalargraph2-1024x409.png)  

![sigmoid](http://solarisailab.com/wp-content/uploads/2016/05/softmax-regression-vectorequation-1024x250.png)
+ 역할  
    * 입력을 sigmoid와 마찬가지로 0과 1사이의 값으로 변환한다.
    * 변환된 결과에 대한 합계가 1이 되도록 만들어준다.

softmax, sigmoid 함수는 [활성화 함수](http://www.aistudy.com/neural/theory_oh.htm#_bookmark_23d2610)

### cross_entropy
+ softmax 모델이 잘 학습하고 있는지에 대한 평가
+ 모델의 예측값이 실제 참값을 설명하는데 얼마나 비효율적인지를 나타냄
    * cross_entropy가 낮을수록 좋은 모델
+ 경사하강법을 이용하여 cross_entropy를 최소화하는 방향으로 최적화시킴

    ![cross_entropy](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-b35ff2c3d573093a809d75e250e35328_l3.png)

### SourceCode
**tensorflow 내부의 학습데이터 가져오기**

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    import tensorflow as tf

본 데이터는 배열 형태의 객체이므로 텐서플로의 convert_to_tensor함수를 이용해 텐서로 변환  

    tf.convert_to_tensor(mnist.train.images)
get_shape 함수로 구조 확인 => (55000,784)  
첫번째 차원은 각 이미지에 대한 인덱스, 두번째 차원은 이미지안의 픽셀수(20x28)  

    print tf.convert_to_tensor(mnist.train.images).get_shape()

 print tf.convert_to_tensor(mnist.train.images).get_shape()
가중치 W와 편향 b를 저장할 변수 생성  
tf.Variable함수를 사용하여 생성되었고 초기값은 모두 0으로 지정

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

x점에 대한 정보를 저장하기 위한 2차원 텐서  
텐서 x는 MNIST이미지를 784개의 실수 벡터로 저장하는 데 사용됨  
None는 어떤 크기나 가능하다는 뜻이며, 여기서는 학습과정에 사용될 총 이미지 개수

    x = tf.placeholder("float", [None, 784])

+ 소프트맥스 함수
    - 0~9까지 각 숫자와 입력 이미지가 얼마나 비슷한지에 대한 확률을 예측하기 위해 사용  


        y = tf.nn.softmax(tf.matmul(x,W) + b)
​    

+ 교차 엔트로피에러
    * 반복이 일어날 때마다 훈련 알고리즘은 훈련 데이터를 받아 신경망에 적용하고 결과를 기댓값과 비교
    * 비용함수를 사용하여 모델이 얼마나 나쁜지를 나타내는 함수를 최소화하는 W와 b를 얻는 것이 목적
    * y의 각 원소 로그값을 구한후, y_의 각 원소를 곱함  
    * tf.reduce_sum함수로 텐서의 모든 원소를 더함
    * 수식


![costfunction](http://solarisailab.com/wp-content/ql-cache/quicklatex.com-b35ff2c3d573093a809d75e250e35328_l3.png)

교차 엔트로피 함수를 구현하기 위해 실제 레이블을 담고있는 새로운 플레이스홀더가 필요함

    y_ = tf.placeholder("float", [None, 10])

위의 y_를 이용해 cross_entropy 비용함수를 구현

    cross_entropy = -tf.reduce_sum(y_*tf.log(y))

+ 경사하강법과 교차 엔트로피 비용함수
    * 샘플에 대한 오차가 계산되면 다음번 루프반복에서는 기댓값과 계산된 값의 차이를 줄이기 위해 모델을 반복적으로 수정해야 함(W,b를 수정)
    * 신경망에서 오차를 후방으로 전파하는 방식인 역전파 알고리즘 사용
        - 가중치 W를 재계산할 때, 출력 값으로부터 언은 오차를 뒤쪽으로 전파
    * 경사하강법과 교차 엔트로피 비용함수를 사용하면 매 루프 반복마다 오차를 줄이기 위해 주어진 상황에서 얼마만큼 매개변수를 변경해야 할지를 계산할 수 있음
    * 학습속도 0.01과 경사 하강법 알고리즘을 사용하여 croee_entropy를 최소화하는 역전파 알고리즘


```python
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
```

모든 변수를 초기화한 후, 세션 시작

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

1000번의 루프를 돌리며, 훈련 데이터셋 100개를 무작위로 추출

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

+ 모델평가
    * tf.argmax(y,1) : 텐서의 한 차원을 따라 가장 큰 값의 인덱스를 리턴
    * 즉, 입력 이미지에 대해 가장 높은 확률을 가진 레이블을 리턴
    * tf.equal을 사용하여 예측값과 실제 레이블을 비교


        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
+ accuracy
    * correct_prediction은 boolean으로 이루어진 리스트 리턴 
* boolean을 수치값으로 변경하여 예측한 것이 얼마만큼 맞는지 확인
     - ex) [true,false,true,true] => [1,0,1,1] => 평균 0.75


        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
