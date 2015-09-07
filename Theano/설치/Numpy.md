#Numpy
[원문](http://deeplearning.net/software/theano/tutorial/numpy.html#matrix-conventions-for-machine-learning)
##Numpy matrix
Numpy를 이용한 행렬 연산<br>

    numpy.asarray([[1., 2], [3, 4], [5, 6]])
    array([[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.]])
    numpy.asarray([[1., 2], [3, 4], [5, 6]]).shape
    (3, 2)
위의 예제는 Numpy를 통해 행렬을 정의하는 코드 예제 <br>
shape() 결과가 3,2로 표시 된것은 3*2 행렬을 표시<br>

<br>
##행렬의 n번째 M열의 항목을 액세스
    numpy.asarray([[1., 2], [3, 4], [5, 6]])[2, 0]
    5.0
위의 코드와 같이 행렬을 정의하고 []안에 행,열 인자 값을 입력하면 해당 값이 출력

#브로드캐스팅
*원문에서 BroadCasting이라고 표기, 한글로 마땅한 단어가 생각 안나 원문 그대로 표기 <br>

    a = numpy.asarray([1.0, 2.0, 3.0])
    b = 2.0
    a * b
    array([2., 4., 6.])

위의 예제는 선형대수에서 행렬에 스칼라 값을 곱하는 역할.<br>
왜 브로드 캐스팅인지 궁금... 

 