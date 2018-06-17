#hadoop 설정

http://rocksea.tistory.com/282
링크의 내용을 복사해서 옮겨 왔습니다.

[HADOOP] 하둡 (Hadoop 2.6.0) 따라하기 설치
2014.12.16 21:03Posted in 프로그래밍 by 광이랑 
위 분 블로그의 내용 캡춰

매번 버젼이 바뀔때 마다 쓰는 것이 지겨워서 한동안 안쓰고 있었는데 예전에 설치하던 시절하고 너무 많이 바껴서 정리를 할 필요가 있겠더군요.
http://rocksea.tistory.com/282
위의 링크는 제자가 열심히 정리한 버젼입니다. 이번 포스트는 저 포스트에서 부족한 부분을 채우는 식으로 정리할려고 합니다.
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

와 같이 추가해 줍니다. 역시나 각각의 cloud0, cloud1, cloud2 에서 작업해줘야 합니다. 주어진 ip 는 제 경우입니다.
1.3 password 없이 각각의 서버에 접속할 수 있게 만들기
master 에서 slave 들에만 접속할 수 있게 만들면 됩니다. 즉 cloud0 -> cloud1 , cloud2 로 연결되어야 하니 cloud0 에서 다음과 같은 작업을 해 줍니다.

$ ssh-keygen -t rsa 

$ ssh cloud@192.168.2.102 mkdir -p .ssh
$ ssh cloud@192.168.2.103 mkdir -p .ssh

$ cat ~/.ssh/id_rsa.pub | ssh cloud@192.168.2.102 'cat >> ~/.ssh/authorized_keys'
$ cat ~/.ssh/id_rsa.pub | ssh cloud@192.168.2.103 'cat >> ~/.ssh/authorized_keys'
1.4 add-apt-repository 을 사용할 수 있게 만들어 주기
우분투(Ubuntu) 를 쓰면서 한번도 add-apt-repository 가 동작 안하는 것을 상상해 본적이 없어서 당황스러웠습니다. 우분투 server 로 설치하면 add-apt-repository 를 사용하기 위해서 필요한 패키지가 있습니다.
$ sudo apt-get install software-properties-common

이것을 설치해 줘야 합니다.
1.5 Java 설치
Java 의 버젼은 상관 없다고들 합니다. 그런데 계속해서 테스트 했던 버젼이 oracle java 이기 때문에 그것을 설치하기로 합니다. 이래저래 귀찮으니까 apt-get 을 이용해서 설치해 주기로 합니다. 뭐니 뭐니 해도 쉬운게 최고입지요.
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
$ sudo apt-get install oracle-java7-installer

질문 나오는 것에 모두 YES 해서 설치해주면 됩니다. 참고로 이건 cloud0, cloud1, cloud2 에서 다 설치해줘야 합니다.
2 하둡 (Hadoop) 설치
Guest OS 세대에 전부 Ubuntu Server 를 설치해주고 네트워크 까지 설치했다면 본격적으로 설치를 시작할 시간입니다. master 로 설정할 cloud0 에서 작업을 시작합니다.
2.1 하둡 다운로드
http://ftp.daum.net/apache/hadoop/core/stable2/hadoop-2.6.0.tar.gz
위의 파일을 다운 로드 받아서 게스트 (Guest) 에 밀어 넣던지 아니면 게스트에서 (인터넷이 된다는 가정하에)
$ cd ~/
$ wget http://ftp.daum.net/apache/hadoop/core/stable2/hadoop-2.6.0.tar.gz

같은 방식으로 다운 받으시면 됩니다. 그리고 rsync 를 이용할 것이기 때문에 master 인 cloud0 에서 작업을 진행하면 됩니다. 최근의 spark 는 2.4.1 대의 하둡을 지원합니다. 그러나 어느날 갑자기 옛날 버젼의 하둡이 사라졌습니다. 현재의 stable2 는 바로 hadoop 2.6.0 버젼입니다. 나중에 spark 는 따로 컴파일을 해줘야 할 것입니다. 아니면 2.6.0 대로 맞춘 버젼이 나오길 기다립시다.
적당한 곳에 풀어줍니다.
$ cd ~/
$ tar xvzf hadoop-2.6.0.tar.gz

$HADOOP_HOME = ~/hadoop-2.6.0 이라고 가정합니다.
2.1.1 실제 데이타 저장 디렉토리 생성
실제로 데이타가 저장될 디렉토리를 만들어 줍니다. cloud0, cloud1, cloud2 에서 모두 만들어줘야 합니다.

$ sudo mkdir -p /data/hadoop/tmp
$ sudo mkdir -p /data/hadoop/dfs/name
$ sudo mkdir -p /data/hadoop/dfs/data

$ sudo chown -R cloud:cloud /data/hadoop

/data/hadoop 의 owner 를 cloud 로 바꿔주는 것을 잊으면 안됩니다. 이 작업을 안해주면 나중에 hadoop 이 포맷하고 실제로 네임노드 , 데이터노드가 실행시 에러가 발생합니다.
2.1.2 $HADOOP_HOME/etc/hadoop/core-site.xml 수정
다음과 같이 수정해줍니다.
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://cloud0:9000</value>
  </property>
  <property>
   <name>hadoop.tmp.dir</name>
   <value>/data/hadoop/tmp</value>
  </property>
</configuration>

fs.defaultFS 는 하둡 파일시스템의 진입점을 보통 지정합니다. master 를 가르키는게 일반적입니다.
2.1.3 $HADOOP_HOME/etc/hadoop/hdfs-site.xml 수정
다음과 같이 수정합니다.
<configuration>
    <property>
        <name>dfs.replication</name>
        <value>2</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:/data/hadoop/dfs/name</value>
        <final>true</final>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:/data/hadoop/dfs/data</value>
        <final>true</final>
    </property>

    <property>
        <name>dfs.permissions</name>
        <value>false</value>
    </property>
</configuration>

dfs.replication 은 HDFS 내에서 몇개의 블록을 복사해서 유지할 것인가 하는 것입니다. 숫자가 커질수록 안정도는 올라가지만 속도는 저하됩니다. 우리는 그냥 2로 정합니다. dfs.namenode.name.dir 과 dfs.datanode.data.dir 은 위에서 지정해 준 디렉토리를 지정해줍니다.
2.1.4 $HADOOP_HOME/etc/hadoop/mapred-site.xml 수정
아마 파일이 없으면 만들어 주시면 됩니다.
<configuration>
  <property>
      <name>mapreduce.framework.name</name>
      <value>yarn</value>
  </property>
  <property>
    <name>mapred.local.dir</name>
    <value>/data/hadoop/hdfs/mapred</value>
  </property>
  <property>
    <name>mapred.system.dir</name>
    <value>/data/hadoop/hdfs/mapred</value>
  </property>
</configuration>

mapreduce.framework.name 이 중요합니다. 최근 하둡은 맵리듀스 관리를 얀(YARN) 으로 합니다. 꼭 지정해 줘야 합니다.
2.1.5 $HADOOP_HOME/etc/hadoop/yarn-site.xml 수정
<configuration>
<!-- Site specific YARN configuration properties -->
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
  <property>
    <name>yarn.nodemanager.aux-services.mapreduce_shuffle.class</name>
    <value>org.apache.hadoop.mapred.ShuffleHandler</value>
  </property>
  <property>
    <name>yarn.resourcemanager.resource-tracker.address</name>
    <value>cloud0:8025</value>
  </property>
  <property>
    <name>yarn.resourcemanager.scheduler.address</name>
    <value>cloud0:8030</value>
  </property>
  <property>
    <name>yarn.resourcemanager.address</name>
    <value>cloud0:8035</value>
  </property>
</configuration>

cloud0 로 쓰여진 부분은 각자 자신의 환경에 맞춰주면 됩니다.
2.1.6 $HADOOP_HOME/etc/hadoop/masters
마스터 (master) 서버가 어떤건지 지정해주는 파일입니다.
cloud0

cloud0 로 지정해주면 됩니다.
2.1.7 $HADOOP_HOME/etc/hadoop/slaves
datanode 로 사용할 서버들을 적어주면 됩니다.
cloud1
cloud2

전체 3대를 사용할 꺼기 때문에 master 를 제외한 나머지를 적어주면 됩니다.
2.1.8 $HADOOP_HOME/etc/hadoop/hadoop-env.sh 수정
export JAVA_HOME=/usr/lib/jvm/java-7-oracle

와 같은 식으로 자바 위치만 정해주면 됩니다.
2.1.9 rsync 를 이용한 복사
cloud0 에서 다음과 같이 실행해 줍니다.
$ rsync -avz ~/hadoop-2.6.0 cloud@cloud1:/home/cloud/
$ rsync -avz ~/hadoop-2.6.0 cloud@cloud2:/home/cloud/

이러면 모든 노드(Node) 의 설정이 동일해집니다. 나중에 따로 생각해 내서 하기 힘들기 때문에 이 부분을 스크립트(script)로 만들어서 노드가 추가되면 스크립트에 추가된 노드분만 추가해 주는 식으로 관리해 주는것이 편할것 같습니다.
2.2 하둡(Hadoop) 실행
사용하기 전에 HDFS 를 포맷해줄 필요가 있습니다.
$ hadoop namenode -format

그리고 분산파일 시스템과 맵리듀스 시스템을 구동시켜주면 됩니다. $HADOOP_HOME 에서
$ sbin/start-dfs.sh
$ sbin/start-yarh.sh

를 실행시켜주면 됩니다. 제대로 동작하고 있는지 확인을 위해서는 cloud0 (즉 마스터) 에서
$ jps

를 입력해서 namenode (dfs 관련 서버) 와 resourcemanager (yarn 관련) 가 구동되어 있는지 확인하면 되고 cloud1 와 cloud2 에서도 마찬가지로
$ jps

를 입력해서 datanode(dfs 관련 서버) 와 nodemanager (yarn 관련)이 구동되어 있는지 확인하면 됩니다.
