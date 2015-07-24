#윈도우에서 Theano 설치 하기 
 
 
##1. python 설치 하기    
Theano를 위해서는 pyCuda와 같은 라이브러리를 컴파일 해야 하기 때문에    일반적인 Python이 아니라    WinPython이라는 패키지로 설치.    이 패키지의 장점은 컴파일 하기 위한 설정이 준비 되어 있다는것.(웹에서 그렇게 말하니 뭐 ^^)   다운로드 위치 : http://winpython.sourceforge.net/
 
 
##2. CUDA 설치  
NVIDIA에서 CUDA 툴킷을 다운로드 받아서 설치.   가장 최근 버젼 : 7.0 버젼   설치시 드라이브가 제대로 설정 안되어 있어서 뭐 어쩌고 저쩌고 메세지 나옴  가볍게 무시하고 설치.   PATH에서 CUDA 경로 추가. 
 
 
 
##3. Visual Studio 설치   
python 패키지 설치를 위해서는 visual Studio나 MinGW 같은 컴파일러가 필요하나...  Visual Studion 2012를 설치.   MS 홈페이지 Visual Studio 2012 Express 버젼을 설치. 
 
 
##4. Visual Studio 일부 수정.    
C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin 디렉토리 밑에    x86_amd64 디렉토리를 복사 amd64로 이름 바꿈.   vcvarsx86_amd64.bat 파일명을 vcvarsx86_amd64.bat로 수정. 
 
 
##5. MinGW 설치   
컴파일을 위해 설치    아래 url에서 다운로드    http://sourceforge.net/projects/mingw-w64/   첨부한 MSYS-20111123.zip파일을 다운로드 받고 설치된 경로에 압축 해재    msys 디렉토리 안에    msys.bat 최상단에    call “c:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\x86_amd64\vcvarsx86_amd64.bat”   라인 추가.       @echo Off 라인 바로 아래에 다음 라인 추가.    set HOME=%USERPROFILE%
 
 
##6. Theano 설치    
pip install Theano로 설치 
 
 
##7. Theano 환경 변수 설정    
c:\User\사용자명\.theanorc.txt 파일 설정  [gcc] 탭에   cxxflags = -shared -I[MinGW 디렉토리]\include -L[파이썬 디렉토리]\libs -lpython34 -DMS_WIN64  외와 같이 설정 
 
 
##8. Theano 설정    
ipython 에서      import theano     theano.test() 
 
   python -c "import theano;theano.test()"
 
   위와 같은 코드가 정상 동작 되면 오케이. 
 
 
참고 사이트 http://rosinality.ncity.net/doku.php?id=python:installing_theano
 
 
이 사이트의 5번 항목은 winPython을 설치하면 굳이 할 필요 없음. 