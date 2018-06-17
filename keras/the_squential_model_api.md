#The Squential Model API
#Keras 순차 모델 API

시작하기 전에 [keras 시작 가이드](./keras_순차모델로_시작하기.md)를 읽어 보세요. 

##useful attributes of model
**model.layes** is a list of the layers added to the model.

#Sequential model methos
#순차 모델 메소드 
###compile 함수

`compile(self, optimizer, loss, metrics=[], sample_weight_model=None)`

###인자값

optimizer : 문자열(옵티마이져의 이름) 혹은 옵티마이져 [객체](./optimizer.md)

loss : 문자열(오브젝티브 함수의 이름) 혹은 오브젝티브 객체 

metrics : 학습 및 테스팅시 사용할 평가 지표. 일반적으로 정확도(accuracy)를 사용 
metrics=['accuracy']

sample_weight_mode:if wou need to do timestep-wise sample weighting(2D weights), set to "temporal".
"None" defaults to sample-wise weights(1D).

kwargs: for Theano backend, ehtese are passed into K. function. ignored for Tensorflow backend. 


###예제 

model = Sqeuential()<br>
model.add(Dense(32,input_shape=(500,)))<br>
model.add(Dense(10,actiovation='softmax'))
model.compile(optimized='rmsprop', 
	loss='categorical_crossentropy',
	metrics=['accuracy'])

  
###fit 함수
`fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)`




