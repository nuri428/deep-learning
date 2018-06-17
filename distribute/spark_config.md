#SPARk 설정

http://rocksea.tistory.com/282
링크의 내용을 복사해서 옮겨 왔습니다.

[HADOOP] 하둡 (Hadoop 2.6.0) 따라하기 설치
2014.12.16 21:03Posted in 프로그래밍 by 광이랑 
위 분 블로그의 내용 캡춰 일부 수정. 

매번 버젼이 바뀔때 마다 쓰는 것이 지겨워서 한동안 안쓰고 있었는데 예전에 설치하던 시절하고 너무 많이 바껴서 정리를 할 필요가 있겠더군요.

1 준비 사항

	* HOST OS: OSX Yosemite
	* 가상 컴퓨터 소프트웨어: VMWare 7.0 (아니면 Virtual Box)
	* Linux Ubuntu 14.04 LTS (Server Version)

1.1 버추얼 박스(Virtual Box)
굳이 버추얼 박스가 아니라도 괜찮습니다. VMWare 나 Parallel 도 괜찮습니다. 버추얼 박스는 공짜기 때문에 제목으로 달아논 것이고 저는 실제로 VMWare Fusion 을 썼습니다.

	* Guest OS 를 우분투로 세개 설치해줍니다. 저는 14.04 LTS 를 이용했습니다. 그리고 세개를 다 깔아주면 불편하기 때문에 한개를 깔고 기본적인 설정을 해주고 그 Guest OS 파일을 복사해주고 Mac Address 를 바꿔주는 식으로 설치해줬습니다.

1.2 네트워크를 고정 아이피로 설정해주기
고정 아이피로 만들면 이후에 세팅에서 엄청나게 편해집니다.
m/shostnameidmastercloud0cloudslavecloud1cloudslavecloud2cloud

위와 같은 형태로 서버들을 구성할 예정입니다. 따라서 다음 부분은 각각의 Guest OS 에 전부 적용해 주어야 합니다.
1.2.1 DHCP 로 받아오는 부분을 static 으로 변경해서 파일을 변경해준다. /etc/network/inteface 를 열어서 다음 부분을 바꾸어 준다.
auto etho0
iface eth0 inet dhcp

이 부분을 커멘트 처리해주고 ('#' 을 맨 앞 라인에 써준다)
auto eth0
iface eth0 inet static 
address 172.16.242.100
netmask 255.255.255.0
gateway 172.16.242.2

와 같은 식으로 적어준다. dhcp 가 static 으로 바뀌고 address , netmask , gateway 를 상황에 맞게 써주는 것을 잊지 않는다. 위의 것은 어디까지 나의 경우에 예에 해당하는 것입니다.
1.2.2 /etc/resolvconf/resolv.conf.d/base 의 수정
예전에는 /etc/resolv.conf 를 수정했으나 이 파일이 이제 서버가 리스타트 될 때마다 리셋이 된다. /etc/resolvconf/resolv.conf.d/base 를 열어서
nameserver 8.8.8.8

를 추가해주고
$ sudo resolvconf -u

로 새로 만들어 주면 된다. 이 작업은 cloud0 , cloud1, cloud2 각각의 상황에 맞게 작성해줘야 합니다.
그리고 /etc/hosts 파일에
172.16.242.100 cloud0
172.16.242.101 cloud1
172.16.242.102 cloud2

spark master를 cloud0로 간주 하고 진행합니다.

와 같이 추가해 줍니다. 역시나 각각의 cloud0, cloud1, cloud2 에서 작업해줘야 합니다. 주어진 ip 는 제 경우입니다.
1.3 password 없이 각각의 서버에 접속할 수 있게 만들기
master 에서 slave 들에만 접속할 수 있게 만들면 됩니다. 즉 cloud0 -> cloud1 , cloud2 로 연결되어야 하니 cloud0 에서 다음과 같은 작업을 해 줍니다.

$ ssh-keygen -t rsa 

$ ssh cloud@192.168.2.102 mkdir -p .ssh
$ ssh cloud@192.168.2.103 mkdir -p .ssh

$ cat ~/.ssh/id_rsa.pub | ssh cloud@192.168.2.102 'cat >> ~/.ssh/authorized_keys'
$ cat ~/.ssh/id_rsa.pub | ssh cloud@192.168.2.103 'cat >> ~/.ssh/authorized_keys'

2.1.4 add-apt-repository 을 사용할 수 있게 만들어 주기
우분투(Ubuntu) 를 쓰면서 한번도 add-apt-repository 가 동작 안하는 것을 상상해 본적이 없어서 당황스러웠습니다. 우분투 server 로 설치하면 add-apt-repository 를 사용하기 위해서 필요한 패키지가 있습니다.
$ sudo apt-get install software-properties-common

이것을 설치해 줘야 합니다.
1.5 Java 설치
Java 의 버젼은 상관 없다고들 합니다. 그런데 계속해서 테스트 했던 버젼이 oracle java 이기 때문에 그것을 설치하기로 합니다. 이래저래 귀찮으니까 apt-get 을 이용해서 설치해 주기로 합니다. 뭐니 뭐니 해도 쉬운게 최고입지요.
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
$ sudo apt-get install oracle-java7-installer

질문 나오는 것에 모두 YES 해서 설치해주면 됩니다. 참고로 이건 cloud0, cloud1, cloud2 에서 다 설치해줘야 합니다.

2 스파크 (Spark) 설치
Guest OS 세대에 전부 Ubuntu Server 를 설치해주고 네트워크 까지 설치했다면 본격적으로 설치를 시작할 시간입니다. master 로 설정할 cloud0 에서 작업을 시작합니다.

2.1 스파크 다운로드
hadoop 없는 링크 http://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-without-hadoop.tgz
hadoop 내장 링크 http://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.6.tgz
위의 파일을 다운 로드 받아서 게스트 (Guest) 에 밀어 넣던지 아니면 게스트에서 (인터넷이 된다는 가정하에)
$ cd ~/
$ wget http://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-without-hadoop.tgz
다른 버젼이 필요할시 [다운로드 페이지](http://spark.apache.org/downloads.html)에서 다른 버젼을 다운로드합니다. 
이 문서에서는 hadoop 2.6이 깔려 있다는 전제 조건하에서 설명합니다.

같은 방식으로 다운 받으시면 됩니다. 그리고 rsync 를 이용할 것이기 때문에 master 인 cloud0 에서 작업을 진행하면 됩니다. 
적당한 곳에 풀어줍니다.
`$ cd ~/`
`$ tar xvzf http://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-without-hadoop.tgz`

`$HADOOP_HOME = ~/hadoop-2.6.0 `
`$SPARK_HOME = ~/spark`

위의 두가지 변수를 가정하고 진행합니다. 

2.1.1 $SPARK_HOME/conf/slaves 작성
$SPARK_HOME/conf/slaves.template를 복사해서 slaves 파일을 만들고 다음과 같이 수정합니다. 
`
cloud0
cloud1
cloud2
---`

2.1.2 $SPARK_HOME/conf/spark_env.sh.template를 복사해서 spark_env.sh 파일을 만들고 다음과 같이 수정합니다. 
`SPARK_MASTER_IP=cloud0
#별도의 HADOOP이 설치 되어 있을시.
HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
`

2.1.3 rsync 를 이용한 복사
cloud0 에서 다음과 같이 실행해 줍니다.
$ rsync -avz ~/spark cloud@cloud1:/home/cloud/
$ rsync -avz ~/spark cloud@cloud2:/home/cloud/

이러면 모든 노드(Node) 의 설정이 동일해집니다. 나중에 따로 생각해 내서 하기 힘들기 때문에 이 부분을 스크립트(script)로 만들어서 노드가 추가되면 스크립트에 추가된 노드분만 추가해 주는 식으로 관리해 주는것이 편할것 같습니다.
2.2 스파크(SPARK) 실행
$ $SPARK_HOME/sbin/start-all.sh

를 실행시켜주면 됩니다. 제대로 동작하고 있는지 확인을 위해서는 cloud0 (즉 마스터) 에서
$ jps

를 입력해서 Worker 와 Master 가 구동되어 있는지 확인하면 되고 cloud1 와 cloud2 에서도 마찬가지로
$ jps

를 입력해서 Worker이 구동되어 있는지 확인하면 됩니다.

정상적으로 동작 여부는 http://cloud:8080에 접속하여 SPARK 화면을 확인 하면 됩니다. 
