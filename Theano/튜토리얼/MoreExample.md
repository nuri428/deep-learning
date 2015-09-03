More Example
------------

이 시점에서 Tensor에 대해 서 공부 하는것이 현명할것이다....

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
At this point it would be wise to begin familiarizing yourself more systematically with Theano’s fundamental objects and operations by browsing this section of the library: Basic Tensor Functionality.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As the tutorial unfolds, you should also gradually acquaint yourself with the other relevant areas of the library and with the relevant subjects of the documentation entrance page.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

Logistic Fucntion
-----------------

로지스틱 함수

![](<943718fb001e2e9576e781d97946d74e44de5251.png>)

이 함수는 도표로 표현하면 다음과 같다.

![](<logistic.png>)

원점을 중심으로 X와 Y축으로 S 자형 커버를 그린다.

 

로지스틱 함수를 코드로 구현하면 다음과 같다.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = function([x], s)
logistic([[0, 1], [-1, -2]])

array([[ 0.5       ,  0.73105858],
       [ 0.26894142,  0.11920292]])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

로지스틱 함수는 많은 성능을 필요로 한다.(나눗셋, 덧셈, 지수함수, 나눗셈)

![](<d27229dfcd1ce305c126bbbd8a2e0fa867ccc503.png>)

위식은 다음과 같이 표현이 가능한다.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = function([x], s2)
logistic2([[0, 1], [-1, -2]])

array([[ 0.5       ,  0.73105858],
       [ 0.26894142,  0.11920292]])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

Computing More than on Thing at the Same Time
---------------------------------------------

Theano 는 Fucntions으로 다수의 아웃풋을 생성하는 것을 지원한다.

예를 들면, matrix a와 b에 대해서 절대값의 차이와 제곱의 차이를 동시에 구할 수
있다.

>   For example, we can compute
>   the [elementwise](<http://deeplearning.net/software/theano/library/tensor/basic.html#libdoc-tensor-elementwise>) difference,
>   absolute difference, and squared difference between two
>   matrices *a* and *b* at the same time

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = function([a, b], [diff, abs_diff, diff_squared])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

>   Note

>   dmatrices produces as many outputs as names that you provide. It is a
>   shortcut for allocating symbolic variables that we will often use in the
>   tutorials.

 

함수 f로 사용하면, 세가지의 리턴값을 가져올 수 있다.

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
f([[1, 1], [1, 1]], [[0, 1], [2, 3]])
[array([[ 1.,  0.],
        [-1., -2.]]),
 array([[ 1.,  0.],
        [ 1.,  2.]]),
 array([[ 1.,  0.],
        [ 1.,  4.]])]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

### Setting a Default Value for an Argument

두개의 인자를 받아 들이는 function이 있다고 할때, 어떤 하나의 변수만을
입력하고자 할때

다른 변수에 대해서 기본값을 설정 할 수 있다.

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from theano import Param
x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, Param(y, default=1)], z)
f(33)
>array(34.0)
f(33, 2)
>array(35.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Param 클래스를 이용하여 function의 입력 파라메터를 정의 할 수 있다.

 

기본값을 이용한 Input은 기본값없이 정의 해야 한다.(like python’s function)

기본값으로 다수의 input을 사용 할 수 있다.

인풋 파라메터들은 일반적인 파이선과 같이 이름으로 포지션을 지정할수 있다.

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = function([x, Param(y, default=1), Param(w, default=2, name='w_by_name')], z)
f(33)
>array(68.0)
f(33, 2)
>array(70.0)
f(33, 0, 1)
>array(33.0)
f(33, w_by_name=1)
>array(34.0)
f(33, w_by_name=1, y=0)
>array(33.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

Note

>   Param does not know the name of the local variables y and w that are passed
>   as arguments. The symbolic variable objects have name attributes (set
>   by dscalars in the example above) and these are the names of the keyword
>   parameters in the functions that we build. This is the mechanism at work
>   in Param(y, default=1). In the case
>   ofParam(w, default=2, name='w\_by\_name'). We override the symbolic
>   variable’s name attribute with a name to be used for this function.

 

Using Shared Variables
----------------------

function 내부 상태를 만들수 있다.

예를 들면, 우리가 가산기를 만든다고 할때,

처음 시작해서는 초기 값은 0이다.

그리고, 각각의 함수가 호출 될때 function의 인자에 따라서 증가 하게된다.

>   It is also possible to make a function with an internal state. For example,
>   let’s say we want to make an accumulator: at the beginning, the state is
>   initialized to zero. Then, on each function call, the state is incremented
>   by the function’s argument.

 

처음으로 가산기 function을 정의한다.

그것은 인자를 내부 상태에 더한다, 그리고 이전 상태를 반환한다.

 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 

이 코드는 몇가지 컨셉을 소개하고 있다.

Shared function 생성자는 Shared Variables를 호출 한다.

hybrid-symbolic과 non-symbolic 변수가  여러개의 function에서 shared(공유) 가능하다. 

공유(shared)변수는 심볼릭 표현(dmatrices(...)와 같은 형태의 객채로 표현 가능)으로 사용 가능하다.
그러나 그들 또한 내부적인 값을 가질수 있으며 그것들은  그 값들이 이러한 심볼릭 변수를  모든  function에서 그렇게 정의 할수 있음을 정의 한다.

그것은 Shared variable(공유변수)라 부른다. 왜냐하면 이 변수의 값은 많은 functions사이에서 공유된다. 
이 Shared variable(공유변수)는 .get_value()메소드를 통해 접근 가능하고, .set_value() 메소드를 통해 수정가능핟. 

이 코드에서 새로운 코드는 functions의 updates 인자 function.updates 이것은 이다. 
 
