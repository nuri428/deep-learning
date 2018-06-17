#openface dnn으로 학습하기 
# pre-trained dnn 모델을 이용한 학습. 
단순한 분류기만을 작성하기 위해서는 이 과정만으로 충분함. 
---
##1 데이터셋 디렉토리 만들기 
<img src="./userimgtree.jpg">
학습 하고자 하는 데이타를 위와 같은 형태의 디렉토리로 작성. 
인식하는 확장자는 jpg, png이며 소문자 여야 함. 
---
##2 전처리
`for N in {1..8}; do ./util/align-dlib.py <path-to-raw-data> align outerEyesAndNose <path-to-aligned-data> --size 96 & done.`
위와 같은 코드를 통해 전처리를 수행. 
align-dlib.py는 분석 예정 
---
##3 representations 제작
`./batch-represent/main.lua -outDir <feature-directory> -data <path-to-aligned-data> creates reps.csv and labels.csv in <feature-directory>.
`
기존의 기계학습기 모델에서 특성(feature 추출)을 추출 하는 과정과 유사. 
batch-represent/main.lua 코드 분석 예정 
---
##4 분류기 모델 작성
`./demos/classifier.py train <feature-directory>`
classifier.py를 통해 분류기 모델을 학습. 
이 코드는 SVM 분류기 모델을 python 피클 파일로 저장. 
1000개 정도의 이미지에서 몇초간의 시간이 소요됨. 
---
##5 새로운 이미지 분류 
`./demos/classifier.py infer ./models/openface/celeb-classifier.nn4.small2.v1.pkl images/examples/{carell,adams,lennon}*`
기존에 작성된 분류기 모델을 이용하여 이미지를 분류 처리 
celeb-classifier.nn4.small2.v1.pkl : 기존 작성된 분류기 모델 
---
이걸로 돌려본 결과.... 별로 안 좋았는데... 
문제는 그게 성능이 안 좋은건지 뭘 몰라서 제대로 못한건지... 
아무래도 후자쪽이 T_T 
