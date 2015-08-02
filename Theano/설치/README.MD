#Theano 설치 관련 자료
윈도우 따로 [이곳](https://github.com/jaeho-kang/deep-learning/blob/master/Theano/%EC%84%A4%EC%B9%98/%EC%9C%88%EB%8F%84%EC%9A%B0%EC%97%90%EC%84%9C%20Theano%20%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0.md)을 통해 확인
[CentOS6](http://deeplearning.net/software/theano/install_centos6.html#install-centos6)
[Ubuntu](http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu)
[Gentoo](http://groups.google.com/d/msg/theano-dev/-8WCMn2FMR0/bJPasoZXaqoJ)

#python으로 Theano 설치 기본 

>pip install Thaeno 

#Python 업데이트 

>pip install --upgrade thaeno
위 명령은 관련 라이브러리 또한 같이 업데이트

>pip install --upgrade --no-deps theano
위 명령은 Theano 라이브러리만 업데이트
scipy, numpy 라이브러리 버젼에서 충돌이 생길수도 있음.

scipy, numpy 라이브러리에서 충돌이 발생이 하는것은 
기존에 설치된 Theano라이브러리 캐시와 이미 설치된 라이브러리가 충돌하는 경우가 있음 
이를 위해 Theano 라이브러리를 다음과 같은 명령으로 정리
원문 
>If you installed NumPy/SciPy with yum/apt-get, updating NumPy/SciPy with pip/easy_install is not always a good idea. This can make Theano crash due to problems with BLAS (but see below). The versions of NumPy/SciPy in the distribution are sometimes linked against faster versions of BLAS. Installing NumPy/SciPy with yum/apt-get/pip/easy_install won’t install the development package needed to recompile it with the fast version. This mean that if you don’t install the development packages manually, when you recompile the updated NumPy/SciPy, it will compile with the slower version. This results in a slower Theano as well. To fix the crash, you can clear the Theano cache like this:

>theano-cache clear

#개발자 버젼 Theano 설치 하기 
stable 버젼(안정 버젼)이 아닌 개발자 버젼을 설치 하고자 할때 다음과 같이 설치 

> pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

또는 git으로 코드를 다운 받아서 python을 통해 직접 설치 가능. 
>git clone git://github.com/Theano/Theano.git
>cd Theano
>python setup.py develo

##주의.
theano setup.py 소스코드는 python 3.x 버젼에 맞지 않음. 
기존의 python2 python3 변환기를 통해 수동으로 변환 시켜야 함. 

#설치 완료 테스트 
>import theano
>theano.test()
위의 코드를 통해 실행 테스트 
주)
CUDA를 이용할 경우 사용하는 그래픽카드가 표시 되고 
중간에 nvcc stdout으로 시작하는 메세지가 같이 표시 되면 정상
중간에 cuda library installed but device not found 메세지가 나오면 
cuda에서 제공하는 nvcc 파일이 있는 패스를 python에서 제대로 인식 못해서 
나오는 에러일 경우가 많음 이런 경우, 패스를 확인

#GPU를 사용하기 
다음과 같은 환경 변수 설정이 필요. 
>CUDA_ROOT
Theano 환경 설정파일에서 cuda 관련 설정 정의 필요. 

#Theano 환경 설정 
사용자 홈 디렉토리 
linux : /home/username/.theanorc
windows : C:\C:\Users\username\.theanorc.txt

해당 파일에 다음과 같은 추가 
>[global]
>device = gpu
>floatX = float32
>[cuda]
>root = CUDA_ROOT 패스(컴퓨터마다 다름)

위의 코드는 저자의 본인 환경(윈도우) 리눅스 일 경우 다른 경로임. 



