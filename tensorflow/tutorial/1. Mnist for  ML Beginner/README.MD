이것은 Tensorflow mnist 예제 입니다. 
여기에서 mnist.py 파일의 94번째 라인을 다음과 같이 수정 해야합니다.

    indices = tf.expand_dims(tf.range(batch_size), 1)
->
    indices = tf.expand_dims(tf.range(0,batch_size), 1)
