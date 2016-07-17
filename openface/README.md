#openface 
- [blog](http://cmusatyalab.github.io/openface/)
- [github](https://github.com/cmusatyalab/openface)
---
#설치
- docker 이용한 설치 
-- `docker pull bamos/openface`
- docker 실행 
-- `docker run -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash`
- [openface with cuda docker image](./Dockerfile)
-- cuda 7.5 cudnn5 ubuntu 14.04
-- nvidia-docker로 실행
- 수작업 설치
--python2 
--dependency
opencv, dlib, numpy, scipy, scikit-learn, scikit-image
--Torch
--dependency
dpnn, nn, optim, csvigo, cutorch or cunn(for cuda), 
fblualib, tds, torchx, optnet : (dnn으로 학습시 필요한 라이브러리)
`for NAME in dpnn nn optim optnet csvigo cutorch cunn fblualib torchx tds; do luarocks install $NAME; done
`

수작업 설치는 좀 버거우니 비추.. 
상기 위의 링크에 있는 Dockerfile을 커스텀 추천
---
# 예제
- 이미지 비교
-- `/root/openface/demos/compare.py images/examples/{lennon*,clapton*}`
- 이미지 인식
-- `./demos/classifier.py infer models/openface/celeb-classifier.nn4.small2.v1.pkl ./images/examples/carell.jpg`
- 웹 캡을 통해 얼굴 인식
-- `./demos/web/start-servers.sh
`

---
#학습
- [pretrain 기반 학습](./train.md)
- [신규 모델 생성](./train_dnn.md)



