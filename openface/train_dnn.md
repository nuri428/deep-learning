#openface dnn으로 학습하기 
# pre-trained dnn 모델을 이용한 학습. 
1. 데이터셋 디렉토리 만들기 
<img src="./userimgtree.jpg">
학습 하고자 하는 데이타를 위와 같은 형태의 디렉토리로 작성. 
인식하는 확장자는 jpg, png이며 소문자 여야 함. 

2. 전처리
`for N in {1..8}; do ./util/align-dlib.py <path-to-raw-data> align outerEyesAndNose <path-to-aligned-data> --size 96 & done.`
위와 같은 코드를 통해 전처리를 수행. 
align-dlib.py는 분석 예정 

3. representations 제작
4. 분류기 모델 작성
5. 새로운 이미지 분류 

# 직접 trainging 학습.
