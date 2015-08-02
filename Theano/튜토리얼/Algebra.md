#대수
[원문](http://deeplearning.net/software/theano/tutorial/adding.html)
원문에 필요해서 일부 주 추가. 

#Tensor
텐서라는 개념을 이해해야 코드가 이해 됨.<br> 
(텐서 개념 소개)[http://ghebook.blogspot.kr/2011/06/tensor.html]<br>
(위키 텐서)[https://ko.wikipedia.org/wiki/%ED%85%90%EC%84%9C]<br>

두 링크가 개념이 어렵긴 마찬가지지만... <br>
그냥 간단히 이야기 하자만 텐서라는 개념은 물리량을 의미 한다.<br>
그리고 이 물리량은 스칼라, 벡터, 행렬이라는 세 물리량의 상위 개념을 의미한다.<br>

python이라는 객체 지향 언어를 이용해서 Theano에서는 tensor라는 클래스를 정의하고, 이 tensor 클래스가 스칼라, 벡터, 행렬이라는 자료형을 일반화시키고 연산자 또한 일반화 시켰다.<br>

#Theano tensor 레퍼런스
[tensor library reference](http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.TensorVariable) <br>
레퍼런스를 보면 알수 있듯이 tensor 객체는 4,5차원 행렬(?)까지도 커버 가능한 타입으로 정의 되어 있다. 
 
#Theano tensor를 이용한 대수
##스칼라 연산 

    import theano.tensor as T
    from theano import function
    x = T.dscalar('x') #float64 스칼라
    y = T.dscalar('y') #float64 스칼라
    z = x + y
    f = function([x, y], z) # f 라는 객체에 x,y를 더해서 결과를 반환하는 함수 객체(functor)를 정의 

    f(2, 3)
    >array(5.0) 
    # 2,3을 더한 결과 5를 표시.
    #array()로 표시 되는 이유는 
    #dscalar()는 0-dimensional arrays(double scalar)로 정의하기 때문에 array로 표기됨.

<br>
##step1
스칼라 변수 정의 

    x = T.dscalar('x')
    y = T.dscalar('y')

    #dscalar를 클래스가 아니라 타입. tensor라는 클래스의 인스턴스의 타입으로 정의됨
    type(x)
    >theano.tensor.var.TensorVariable
    x.type
    >TensorType(float64, scalar)
    T.dscalar
    >TensorType(float64, scalar) 
    x.type is T.dscalar
    >True
 
<br>
##step2
x와 y와를 더하여 z에 결과를 할당

    z = x + y
    from theano import pp
    print pp(z)
    > (x + y)
      
<br>
##step3
x와 y를 더하여 z에 할당, 이것을 함수로 정의 

    f = function([x,y],z)


##step4
f활용

    f(2,3)
    >array(5.0)
    f(16.3, 12.1)
    >array(28.4)
    #.eval() 함수를 정의하지 않고 eval()함수를 통해 비슷한 결과를 가져올 수 있음.
    #대신 eval()은 function객체보다 활용면이 조금 떨어짐.
<br>
As a shortcut, you can skip step 3, and just use a variable’s eval() method. The eval() method is not as flexible as function() but it can do everything we’ve covered in the tutorial so far. It has the added benefit of not requiring you to import function() . Here is how eval() works:
<br>

    
#행렬 연산 
    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x + y
    f = function([x, y], z)
    >array([[ 11.,  22.], [ 33.,  44.]])
    #f 함수에 두개의 인자를 입력하면 두 인자의 합을 리턴
<br>

numpy를 이용한 f 활용

    import numpy
    f(numpy.array([[1, 2], [3, 4]]), numpy.array([[10, 20], [30, 40]]))
    >array([[ 11.,  22.],
       [ 33.,  44.]])
    #numpy 객체를 이용해서도 f 함수를 사용할 수 있음.


#벡터 연산
벡터 객체를 이용한 함수 선언

    import theano
    a = theano.tensor.vector() # declare variable
    out = a + a ** 10               # build symbolic expression
    f = theano.function([a], out)   # compile function
    print f([0, 1, 2])  
    >array([0, 2, 1026])
    
#Graph
함수 선언과 그래프<br>
function 정의는 그래프로 표시 가능하다. <br>

    x = dmatrix('x')
    y = dmatrix('y')
    z = x + y
##위의 코드는 아래와 같은 그래프를 의미한다.<br>

<img src="graph-apply.png">

위의 이미지에서 <br>
파란상자는 apply 노드를 의미 한다.<br>
빨간상자는 변수 노드를 의미 한다.<br>
녹색원은 함수 노드를 의미 한다.<br>
보라상자는 타입을 의미 한다.<br>

>원문<br>
>Arrows represent references to the Python objects pointed at. The blue box is an Apply node. Red boxes are Variable nodes. Green circles are Ops. Purple boxes are Types.
When we create Variables and then Apply Ops to them to make more Variables, we build a bi-partite, directed, acyclic graph. Variables point to the Apply nodes representing the function application producing them via their owner field. These Apply nodes point in turn to their input and output Variables via their inputs and outputs fields. (Apply instances also contain a list of references to their outputs, but those pointers don’t count in this graph.)
The owner field of both x and y point to None because they are not the result of another computation. If one of them was the result of another computation, it’s owner field would point to another blue box like z does, and so on.
Note that the Apply instance’s outputs points to z, and z.owner points back to the Apply instance.

<br>
다음의 두 코드는 두개의 변수를 입력 받아 곱한 결과를 만드는 그래프를 생성하는 코드이다.<br> 

앞의 코드는 짧은 예제이고,<br> 
뒤의 코드는 긴 예제이다.<br> 
대부분의 코드는 뒤의 예제와 같은 양식으로 구현되어 있다. (정리자 주)<br>

    #short example
    # create 3 Variables with owner = None
    x = T.matrix('x')
    y = T.matrix('y')
    z = T.matrix('z')

    # create 2 Variables (one for 'e', one intermediate for y*z)
    # create 2 Apply instances (one for '+', one for '*')
    e = x + (y * z)

<br>

    #long example
    from theano.tensor import add, mul, Apply, Variable, TensorType

    # Instantiate a type that represents a matrix of doubles
	# 필요한 객체 타입 선언. float64, double, broadcastable, matrix 속성 설정.
    float64_matrix = TensorType(dtype = 'float64',              # double
                            broadcastable = (False, False)) # matrix

    # We make the Variable instances we need.
    # 객체 타입을 기반으로 객체 인스턴스 생성,
    x = Variable(type = float64_matrix, name = 'x')
    y = Variable(type = float64_matrix, name = 'y')
    z = Variable(type = float64_matrix, name = 'z')

    # This is the Variable that we want to symbolically represents y*z
    # y*z 그래프 정의 
    # 오퍼레이터를 당당할 객체 선언
    mul_variable = Variable(type = float64_matrix)
    assert mul_variable.owner is None

    # Instantiate a symbolic multiplication
    # 곱셈을 담당할 노드 정의
    node_mul = Apply(op = mul,   # 오퍼레이터 정의
                 inputs = [y, z],# 입력인자 정의
                 outputs = [mul_variable])#아웃풋 정의
    # Fields 'owner' and 'index' are set by Apply
    assert mul_variable.owner is node_mul
    # 'index' is the position of mul_variable in mode_mul's outputs
    assert mul_variable.index == 0

    # This is the Variable that we want to symbolically represents x+(y*z)
    # x + ( y * z ) 정의
    add_variable = Variable(type = float64_matrix)
    assert add_variable.owner is None

    # Instantiate a symbolic addition
    node_add = Apply(op = add,
                 inputs = [x, mul_variable],
                 outputs = [add_variable])
    # Fields 'owner' and 'index' are set by Apply
    assert add_variable.owner is node_add
    assert add_variable.index == 0

    e = add_variable

    # We have access to x, y and z through pointers
    assert e.owner.inputs[0] is x
    assert e.owner.inputs[1] is mul_variable
    assert e.owner.inputs[1].owner.inputs[0] is y
    assert e.owner.inputs[1].owner.inputs[1] is z

#Automatic wrapping

     e = dscalar('x') + 1
<br>
위의 코드는 아래와 같이 자동으로 랩핑 된다. 

    node = Apply(op = add,
    inputs = [Variable(type = dscalar, name = 'x'),
                       Constant(type = lscalar, data = 1)],
    outputs = [Variable(type = dscalar)])
    e = node.outputs[0]




