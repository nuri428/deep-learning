#SPARK
#이렇게 코딩 하면 안된다. 
##case1
a = aDF.collect()
b = bDF.collect()

위의 코드가 호출되면 a,b의 값은 분산된 각각의 executor에서 driver로 모여서 collect()을 수행후 할당된다. 

만약 aDF, bDF가 매우 크다면 
Driver는 Out Of Memory(OOM)에러를 발생,
그리고 각각의 분산된 executor에서 driver로 데이타를 전송, 재 조립에 많은 시간을 소요한다. 

##case2
cDF = sqlContext.createDataFrame(a+b)
cDF라는 새로운 프레임을 생성하고 그 데이타를 Driver에서 Executor로 전송.
만약 cDF가 엄청 큰 데이타라면 
Driver는 Out Of Memory(OOM)에러를 발생,
그리고 각각의 분산된 executor에서 driver로 데이타를 전송, 재 조립에 많은 시간을 소요한다. 

## 해결 방법
cDF = aDF.unionAll(bDF)
DataFrame의 unionAll() 메소드를 이용
이 코드는 Executor에서만 수행된다. 
다시 말하면 Driver에서 데이타를 각각의 Executor에서 가져오고 다시 조립하여 다시 Executor로 보내는 작업을 수행 하지 않음. 

##성능향상을 위한 팁
take(n)을 대체하기 위해서 절대 collect()를 쓰지 마라. 
자주 사용하는 DataFrame 객체는 cache() 메소드를 이용하라.
cache()는 DataFrame 객체를 자주 사용하므로 캐쉬에 올리라는 명령어. 

#프로젝트 텅스텐
https://databricks.com/blog/2015/04/28/project-tungsten-bringing-spark-closer-to-bare-metal.html

java는 모든 데이타를 객체로 저장하여 메모리 오버헤드가 많음.
이 부분을 텅스텐 프로젝트를 통해 매니징하여 메모리 오버헤드를 줄이고 연산의 최적화를 꾀한다. 
1.4 기준.
2.

