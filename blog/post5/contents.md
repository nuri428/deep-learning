#tensorflow 학습시 데이타가 밸런스가 맞지 않을때 처리 방법 

두개의 클래스(정상/불량, 0/1) 데이터를 지도 학습 시킬때

cross_entropy 함수는 대부분 reduce_mean()을 통해 학습을 시키는데 

클래스0 대 클래스1 의 데이타 비율이 1:4 일 경우 

cross_entropy 계산시 편향된 결과값이 나타나게 됩니다. 

이럴 cross entropy를 다음과 같은 방법을 이용하면 해결이 가능하다고 합니다. 



일반적인 경우 

```python
cross_entropy = -tf.reduce_mean(ys * tf.log(predict + 1e-12))
```

클래스 데이타 밸런스가 안 맞는 경우 

```python
classes_weights = tf.constant([0.1, 0.9])
        cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
            logits=predict, targets=ys, pos_weight=classes_weights))
```



그외 다른 여러 가지 방법은 다음 [링크](https://stackoverflow.com/questions/35155655/loss-function-for-class-imbalanced-binary-classifier-in-tensor-flow)를 참고 하세요 

