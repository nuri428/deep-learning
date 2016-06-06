#Optimizer

#Useage of optimizers 

An optimizer is one of the two arguments required for compiling a Keras model:
옵티마이져는 Keras 모델을 컴파일 하기 위한 두가지 인자 값 중 하나. 

`
<br>
model = Sequential()<br>
model.add(Dense(64, init='uniform', input_dim=10))<br>
model.add(Activation('tanh'))<br>
model.add(Activation('softmax'))<br>

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)<br>
model.compile(loss='mean_squared_error', optimizer=sgd)<br>
<br>
`

위의 예에서 보는 것과 같이 model.compile()을 호출할때 최적화된 인스턴스를 전달하거나, 인스턴스의 이름을 콜 함으로써 인수를 전달 할 수 있다. <br>
ps.<br>
model.compile()메소드에서 옵티마이져의 인스턴스를 생성하여 전달하거나, 이미 정의되어 있는 옵티마이저의 이름을 전달함으로써 사용할 수 있다. <br>
뒤의 경우(이미 정의 되어 있는 옵티마이저의 이름을 전달)기본 인자값들이 전돨되어 사용된다. 

`
pass optimizer by name; default parametes will be used 
model.compile(loss='mean_squared_error', optimized='sgd')
<br>
`

ps. ADA 관련 정보는 아래 링크 참고 
[Advanced Gradient Descent Method](http://newsight.tistory.com/224)

#SGD

`
keras.optimizeds.SGD(lr=0.01, momentum=0.0, decay=00, nestrov=False)
`

(Stochastic gradient descent)확률적 경사 하강법은 모멘텀(momentum), 학습률(learning rate), 붕괴(decay), nesterov값을 인자로 받는다. 

##인자값
lr : 실수, 0 >= 실수, 학습률<br>
momentum : 실수, 0 >= 실수, 파라메타 갱신 모멘텀(관성값)<br>
decay : 실수, 0 >= 실수, Learning rate over each update.<br>
nestrov : boolean, Whether to apply Nestrov momentum <br>

ps. 관성(모멘텀)은 현단계에서 계산된 gradient값을 사용하는 대신 이전단계에서 사용된 gradient 값을 일정한 %만큼 반영하여 새로 게산된 gradient와 합해서 사용한다. 
즉 원래의 gradient를 일정 부분 유지하면서 새로운 gradient를 적용하여, 관성 모멘트 같은 효과를 주는 방식. 

#RMSprop
`
keras.optimizers.RMSprop(lr=0.001, rhs=0.9, epsilon=1e-08)
`

RMSProp optimizer.

It is recommended to leave the parameters of this optimizer at their default values(except the learning rate, which can be freely tuned).

This optimizer is usually a good choice for recurrent neural networks.

##인자값
lr : 학습률, 실수, 0<= lr<br>
rho : rhs, 실수, 0<= rho<br>
epsilon : 입실론, 실수, 0<=, Fuzz factor<br>


#Adagrad
`
keras.optimized.Adagrad(lr=0.01, opsilonn=1e-08)
<br>
`

Adagrad optimizer. <br>
It is recomended to leave the parametes of this optimizer at their default values.<br>

##인자값
lr : 학습률, 실수, 0<= lr<br>
epsilon : 입실론, 실수, 0<= epsilon<br>

ps. Adagrad는 학습률(lr)값을 L2 노말라이즈(평준화)값으로 나눈값. 


#Adadelta
`
keras.optimizers.Adadelta(lr=1.0, rhs=0.95, epsilon=1e-08)
<br>
`

Adadelta optimizer.<br>
It is recommended to leave the parameters of this optimizer at their default values. <br>

##인자값
lr : 학습률, 실수, 0<= lr, It is recommended to leave it at the default value.<br>
rhs : rhs,  실수, 0<= rho<br>
epsilon : 입실론, 실수, 0>= epsilon, Fuzz factor <br>

ps. Adadelta는 학습률(lr)을 RMS값으로 나눈 형태 

참고 :[Adadelta - an adaptive learning rate method.](http://arxiv.org/abs/1212.5701)

#Adam
`
<br>
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.001, epsilon=1e-08)
<br>
`

Adam optimizer.

기본 파라메터는 해당 알고리즘을 제안한 문서의 값을 따른다. 

##인자값
lr : 학습률,실수, 0 >= lr<br>
beta_1/beta_2 : ,실수, 0 < beta < 1. Generally close to 1. <br>
epsilon : 입실론,실수, 0 <= epsilon, Fuzz.factor.<br>

참고 :[Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

#Adamax
`
<br>
keras.optimizers.admax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
<br>
`

Adamax 최적화는 Adam의 논문 섹션 7이 출처. 
It is a variant of Adam based on the infinity norm.
기본 파라메터는 해당 알고리즘을 제안한 문서의 값을 따른다. 

##인자값
lr : 학습률, 실수, 0>= lr<br>
beta_1/beta_2 : 베타, 실수, 0 < beta < 1. Generally close to 1. <br>
epsilon : 입실론, 실수, 0<= epsilon, Fuzz factor<br>

참조 :[Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)


