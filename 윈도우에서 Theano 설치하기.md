#윈도우에서 Theano 설치 하기 
 
 
##1. python 설치 하기    
Theano를 위해서는 pyCuda와 같은 라이브러리를 컴파일 해야 하기 때문에일반적인 Python이 아니라
WinPython이라는 패키지로 설치.    
이 패키지의 장점은 컴파일 하기 위한 설정이 준비 되어 있다는것.(웹에서 그렇게 말하니 뭐 ^^)   
다운로드 위치 : http://winpython.sourceforge.net/
 
 
##2. CUDA 설치  
NVIDIA에서 CUDA 툴킷을 다운로드 받아서 설치.   
 7.0 버젼   설치시 드라이브가 제대로 설정 안되어 있어서 뭐 어쩌고 저쩌고 메세지 나옴  가볍게 무시하고 설치.   
 PATH에서 CUDA 경로 추가. 
 
 
 
##3. Visual Studio 설치   
python 패키지 설치를 위해서는 visual Studio나 MinGW 같은 컴파일러가 필요하나...  
Visual Studion 2012를 설치.   
MS 홈페이지 Visual Studio 2012 Express 버젼을 설치. 
 
 
##4. Visual Studio 일부 수정.    
C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin 디렉토리 밑에 x86_amd64 디렉토리를 복사 amd64로 이름 바꿈.   vcvarsx86_amd64.bat 파일명을 vcvarsx86_amd64.bat로 수정. 

##5. boost라이브러리 설치 
cuda 및 pycuda에서 일부 사용하기 때문에 필수적으로 설치 
[boost](http://www.boost.org/)에서 최근 버젼을 다운로드 받고 설치 
###설치 과정 
임의의 경로에 파일을 풀고 "bootstrap.bat"파일 실행 
비주얼 스튜디오가 정상적으로 설치 되었다면 "b2.exe"파일 생성
"b2.exe"파일을 실행시키면 boost정상 설치 
[부스트 홈페이지](http://www.boost.org/doc/libs/1_55_0/more/getting_started/windows.html)참고
 
##6. MinGW 설치   
컴파일을 위해 설치    아래 url에서 다운로드    http://sourceforge.net/projects/mingw-w64/   첨부한 [MSYS-20111123.zip](http://sourceforge.net/projects/mingw-w64/files/External%20binary%20packages%20%28Win64%20hosted%29/MSYS%20%2832-bit%29/MSYS-20111123.zip/download)파일을 다운로드 받고 설치된 경로에 압축 해재    
msys 디렉토리 안에    
msys.bat 최상단에    
call “c:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\x86_amd64\vcvarsx86_amd64.bat”   라인 추가.
@echo Off 라인 바로 아래에 다음 라인 추가.    
set HOME=%USERPROFILE%

위와 같은 내용을 적용후 msys.bat 실행
커맨드 라인이 뜨면 
"sh /postinstall/pi.sh" 명령 실행 
mingw가 설치 되었는지 물어보고, 
설치 경로를 설정하라고 함.

##7. pycuda 설치 
pip install pycuda로 설치 
boost가 설치 안되어 있을 경우 설치 오류 발생. -> boost 설치 
컴파일러 경로 오류시 오류 발생 -> 6번 항목에서 mingw 설치 경로 입력 부분 수정 

##8. Theano 설치    
pip install Theano로 설치 
 
##9. Theano 환경 변수 설정    
c:\User\사용자명\.theanorc.txt 파일 설정  [gcc] 탭에   cxxflags = -shared -I[MinGW 디렉토리]\include -L[파이썬 디렉토리]\libs -lpython34 -DMS_WIN64  외와 같이 설정 
 
 
##10. Theano 설정    
ipython 에서      import theano     theano.test() 
 
   python -c "import theano;theano.test()"
 
   위와 같은 코드가 정상 동작 되면 오케이. 
 
 
[참고 사이트](http://rosinality.ncity.net/doku.php?id=python:installing_theano)
주 winpython을 이용할 경우 참고 사이트에 적힌 5번 항목은 굳이 할 필요 없음. 
