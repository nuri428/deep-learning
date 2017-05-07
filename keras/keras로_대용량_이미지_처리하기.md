#Keras로 대용량 이미지 처리하기
keras에 있는 샘플들은 용량이 작아서 기존의 data_load()함수를 오버랩핑해서 처리 해도 별 문제가 없습니다. 

하지만, 대용량 이미지 데이터의 경우(저자의 경우 이미지 파일이 100기가 정되되는 데이타를 학습 시킬 예정입니다 ^^) 이 모든 이미지들을 메모리에 올리려고 했다가는 바로 메모리 용량 부족 메세지 나옵니다. 

메모리 용량을 올려서 처리해도 메모리 용량보다 큰 이미지 데이타셋일 경우 처리가 어렵습니다. 

이를 위해서 kears에서는 이미지 데이터제너레이터와 제너레이터를 통한 학습을 제공하고 있습니다. 

---

#주의사항
아래 내용을 이용하기 위해서는 라이버러리 버젼이 다음과 같아야 합니다. 

keras==1.0.4<br>
numpy >= 1.10

---

##ImageDataGenerator
이 클래스의 flow_from_directory 함수를 통해 데이터제너레이터를 할 수 있습니다. 
자세한 사용법은 따로 글을 올리곘습니다. 

```
gen = ImageDataGenerator().flow_from_directory(
        sys.argv[1],
        target_size=(input_width, input_height),
        batch_size=32,
        class_mode='categorical')
```

flow_from_directory인자는 크게 네가지가 필요로 합니다. 

###directory
  이미지 데이터가 저장되어 있는 경로

경로/클래스1/이미지1<br>
경로/클래스1/이미지2<br>
경로/클래스2/이미지1<br>
,,,
경로/클래스n/이미지n<br>

이런식으로 디렉토리 구조를 만들면 클래스와 이미지를 자동을 인식 합니다. 

###target_size
>이미지의 크기를 지정 합니다.<br> 
(width, height)


###batch_size
>한번에 처리를 해야할 데이타의 양을 설정 합니다. 

###class_mode 
>데이터의 라벨 타입을 설정 합니다. <br>
binary,categorical,sparse <br>
셋중 하나를 설정 합니다. 

---

##학습
###model.fit_generator
>generator을 데이타로 입력 받으면 model.fit함수가 아니라 <br>
model.fit_generator을 사용합니다. 

fit_generator함수로 세가지 인자를 기본으로 사용합니다. 

###generator 
>위의 이미지제너레이터에서 작성한 제너레이터를 인자로 받습니다. 

###samples_per_epoch
>epoch당 처리하는 샘플의 갯수 입니다. 

###nb_epoch
>epoch 횟수를 설정합니다. 

[예제코드](./vgg-sequence-learn.py)
