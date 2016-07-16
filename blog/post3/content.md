#openFace api 사용기 

얼굴 인식 알고리즘 중 가장 최근에 발표된 내용 

{논문}[http://www.cv-foundation.org/openaccess/content_cvpr_2015/app/1A_089.pdf]

{오픈페이스 사이트}[https://cmusatyalab.github.io/openface/]

트리플셋을 기반으로 얼굴인식 알고리즘. 

##며칠 써본 느낌.. 
1. 도커로 배포해서 환경 설정 삽질 안해도 된다 (우와~, 다만 도커 개념이 없어서 해맴....)

2. 학습을 시키기 위한 첫번째 작업에서 이미지를 엄청 줄여버린다. 
   왜 줄이는지 논문이나 코드를 좀 읽어 봐야 할 듯..
   이게 자동으로 되는지 몰라서 어제(7월 15일) 데이터 정제 삽입 했는데... 

3. 이 코드를 이용한 예제로 빅뱅이론에서 두 남자 주인공을 동영상에서 트랙킹 하는 예제를 보니 
   나도 저런가 만들고 싶어서 열심히 삽질을.. ㅋ 

4. cnn은 역시 오래 걸린다.... 
   지금 돌리는 머신에는 쿠다 지원 그래픽 카드가 없어서 시피유 모드로 돌리는데 오래 걸림... 

5. 라이버러리가 좀 햇깔림.. 
   파이선, 루아 두 언어로 작성. 
   루아왜 내부를 만들고, 파이썬으로 감싼 듯.
   아직 까지 파이썬으로 다 커버 못해서 루아를 이용해야 하는 부분이 보임. 

7월 16일 오후 3시... 
학습 돌리고 스터디 하러... 

dnn 학습 도중 오류 발생. 

`
/root/torch/install/bin/luajit: /root/torch/install/share/lua/5.1/trepl/init.lua:384: module 'cutorch' not found:No LuaRocks module found for cutorch
	no field package.preload['cutorch']
	no file '/root/.luarocks/share/lua/5.1/cutorch.lua'
	no file '/root/.luarocks/share/lua/5.1/cutorch/init.lua'
	no file '/root/torch/install/share/lua/5.1/cutorch.lua'
	no file '/root/torch/install/share/lua/5.1/cutorch/init.lua'
	no file './cutorch.lua'
	no file '/root/torch/install/share/luajit-2.1.0-beta1/cutorch.lua'
	no file '/usr/local/share/lua/5.1/cutorch.lua'
	no file '/usr/local/share/lua/5.1/cutorch/init.lua'
	no file '/root/.luarocks/lib/lua/5.1/cutorch.so'
	no file '/root/torch/install/lib/lua/5.1/cutorch.so'
	no file '/root/torch/install/lib/cutorch.so'
	no file './cutorch.so'
	no file '/usr/local/lib/lua/5.1/cutorch.so'
	no file '/usr/local/lib/lua/5.1/loadall.so'
stack traceback:
	[C]: in function 'error'
	/root/torch/install/share/lua/5.1/trepl/init.lua:384: in function 'require'
	./training/main.lua:16: in main chunk
	[C]: in function 'dofile'
	/root/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:145: in main chunk
	[C]: at 0x00406670
`

cutorch가 설치 되어 있지 않아서 발생...
docker에서는 다 되는줄 알았는데... 

`
update torch first: 
luarocks install torch
luarocks install cutorch
`

docker에서 nvidia 디바이스를 지정해서 연결 해줘야 cuda를 제대로 씀. 


> ls -la /dev | grep nvidia   <- 이 커맨드로 디바이스 찾음.
> crw-rw-rw-  1 root root    195,   0 Oct 25 19:37 nvidia0 
> crw-rw-rw-  1 root root    195, 255 Oct 25 19:37 nvidiactl
> crw-rw-rw-  1 root root    251,   0 Oct 25 19:37 nvidia-uvm


위 명령어로 나온 디바이스를 docker에 연동 시킴.

`--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm`

새벽 4시 cutorch 컴파일중... 


