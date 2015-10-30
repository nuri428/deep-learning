More Example
------------

이 시점에서 Tensor에 대해서 공부 하는것이 현명할것이다....

At this point it would be wise to begin familiarizing yourself more systematically with Theano’s fundamental objects and operations by browsing this section of the library: Basic Tensor Functionality.

As the tutorial unfolds, you should also gradually acquaint yourself with the other relevant areas of the library and with the relevant subjects of the documentation entrance page.

 

Logistic Fucntion
-----------------

로지스틱 함수

![](<943718fb001e2e9576e781d97946d74e44de5251.png>)

이 함수는 도표로 표현하면 다음과 같다.

![](<logistic.png>)

원점을 중심으로 X와 Y축으로 S 자형 커버를 그린다.

로지스틱 함수를 코드로 구현하면 다음과 같다.

	x = T.dmatrix('x')
	s = 1 / (1 + T.exp(-x))
	logistic = function([x], s)
	logistic([[0, 1], [-1, -2]])

	>array([[ 0.5       ,  0.73105858],
       [ 0.26894142,  0.11920292]])
 

로지스틱 함수는 많은 성능을 필요로 한다.(나눗셋, 덧셈, 지수함수, 나눗셈)

![](<d27229dfcd1ce305c126bbbd8a2e0fa867ccc503.png>)

위식은 다음과 같이 표현이 가능한다.

	s2 = (1 + T.tanh(x / 2)) / 2
	logistic2 = function([x], s2)
	logistic2([[0, 1], [-1, -2]])

	>array([[ 0.5       ,  0.73105858],
       [ 0.26894142,  0.11920292]])
 

Computing More than on Thing at the Same Time
---------------------------------------------

Theano 는 Fucntions으로 다수의 아웃풋을 생성하는 것을 지원한다.

예를 들면, matrix a와 b에 대해서 element wise 절대값의 차이와 제곱의 차이를 동시에 구할 수
있다.

>   For example, we can compute
>   the [elementwise](<http://deeplearning.net/software/theano/library/tensor/basic.html#libdoc-tensor-elementwise>) difference,
>   absolute difference, and squared difference between two
>   matrices *a* and *b* at the same time

	a, b = T.dmatrices('a', 'b')
	diff = a - b
	abs_diff = abs(diff)
	diff_squared = diff**2
	f = function([a, b], [diff, abs_diff, diff_squared])
 

>   Note

>   dmatrices produces as many outputs as names that you provide. It is a
>   shortcut for allocating symbolic variables that we will often use in the
>   tutorials.

 

함수 f로 사용하면, 세가지의 리턴값을 가져올 수 있다.

	f([[1, 1], [1, 1]], [[0, 1], [2, 3]])
	[array([[ 1.,  0.],
        [-1., -2.]]),
	 array([[ 1.,  0.],
        [ 1.,  2.]]),
	 array([[ 1.,  0.],
        [ 1.,  4.]])]
 

### Setting a Default Value for an Argument

두개의 인자를 받아 들이는 function이 있다고 할때, 어떤 하나의 변수만을
입력하고자 할때

다른 변수에 대해서 기본값을 설정 할 수 있다.

	from theano import Param
	x, y = T.dscalars('x', 'y')
	z = x + y
	f = function([x, Param(y, default=1)], z)
	f(33)
	>array(34.0)
	f(33, 2)
	>array(35.0)


Param 클래스를 이용하여 function의 입력 파라메터를 정의 할 수 있다.


기본값을 이용한 Input은 기본값 없이 정의 해야 한다.(like python’s function)

기본값으로 다수의 input을 사용 할 수 있다.

인풋 파라메터들은 일반적인 파이선과 같이 이름으로 포지션을 지정할수 있다.

 

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
이 Shared variable(공유변수)는 .get_value()메소드를 통해 접근 가능하고, .set_value() 메소드를 통해 수정가능하다. 

이 코드에서 새로운 코드는 functions의 updates 인자 function.updates는 반드시 리스트 쌍을 지원해야 한다.(공유변수, 새로운 변수) 

it can also be a dictionary whose keys are shared-variableds and values are the new expressions
그것은 키와  공유 변수 그리고 변수들로 구성된 새로운 표현식으로 사전과 같아 진다. 

Eitger way. it means "wherevever this function runs, it will replace the .value of each shared variable with the result of the corresponding expression". 

같은 방법으로 그것은 다음을 의미한다. " 언제든 이 function이 수행 될때, 그것은 .value를 각각의 올바른 표현의 결과로서 공유 변수로 대체 한다."

Above, our accumulator replaces the state's value with the sum of the state and increment amount. 
위의 예에서 우리의 덧셈기(accumulator)는  상태의 변수를 상태와 더해진 결과로 대체한다.  

다음과 같은 예제를 보면 테스트 해보자. 


> state.get_value()
<br>array(0)
> <br>accumulator(1)
<br>array(0)
> <br>state.get_value()
<br>array(1)
> <br>accumulator(300)
<br>array(1)
> <br>state.get_value()
<br>array(301)
 
 set_value() 메소드를 통해 상태를 재설정 하는것도 가능하다. 
 >state.set_value(-1)
 <br>accumulator(3)
 <br>array(-1)
 <br>>state.get_value()
 <br>array(2)
 

 As we mentioned above, you define more than one function to use the same shared variable. These functions can all update the value. 
 <br>위에서 언급했듯이, 당신은 여러개의 function에서 같은 공유 변수를 정의 할 수 있다. 그 function들은 모두 값을 갱신(update)할 수 있다. 
 >decrementor = function([inc], state, updates=[(state, state-inc)])
 <br>decrementor(2)
 <br>>array(2)
 <br>state.get_valye()
 <br>>array(0)
 
 You might be wondering why updates mechanism exists.
 You can always achieve a similar result by returning the new expressions, and working with them in NumPy as usual. 
 The updates mechnism can be a syntactic convenience, but it is mainly there for efficiency. Updates to shared variabled can sometimes be done more quickly using in-place algorithm. (e.g. low-rank matrix updates). Also, Theano has more control over where and how shared variables are allocated, which is one of the important elements of getting good performatce on GPU.
 <br>
 It may happen that you expressed some formula using a shared variable, but you do not want to use its value. In this case, you can use the givens parameter of function which replace a particular node in a graph for the purpose of one particular function.
 <br>
 <br>
 update 메카니즘이에 대해서 주의하라. 당신은 또한 새로운 표현법을 통해서 쉽게 결과를 가져올수 있다. 그리고, 그러한것을 NumPy를 이용해서도 같은 방법으로 사용할 수 있다. update 매커니즘은 문법적인 변환이 가능하다. 그러나 그것은 효율적이지 않다. 공유 변수의 갱신은 때때로 내부 알고리즘보다 빨리 이루어 질 수 있다. (예를 들면 low-rank matrix 갱신과 같은(?)). 또한, Theano 공유 변수가 어디에 있는지 그리고 어떻게 공유 변수를 할당 하는 방법을 제공한다. 그것은 GPU를 이용한 좋은 성능을 가져오는 중요한 요소중 하나이다. 
<br>
공유 변수를 이용하여 일부 식을 표현하고, 변수처럼 사용하지 않을 수 있다. 이 경우, 하나의 특정 function의 목적 그래프에 특정노드를 대체함수의 주어진 매개변수를 사용할 수 있다.  
 
	fn_of_state = state * 2 + inc
	#The type of foo myst match the shared variable we are replaceing with the givens
	foo = T.scalar(dtype=state.dtype)
	skip_shared = function([inc, foo], fn_of_state,
                           givens=[(state, foo)])
	skip_shared(1, 3) # we skip using 3 for the state, not state.value
	array(7)
	state.get_value() # old state still there, byt we didn't use it
	array(0)


<br>
The givens parameter can be used to replace any symbolic variable, not just a shared variable. You can replace constants, and expressions, in general. Be careful though, not to allow the expressions introduced by a givens substitution to be co-dependent, the order of substitution is not defined, so the substitutions have to work in any order. <br>
In practice, a good way of thinking about the givens is as a mechanism that allows you to replace any part of your formula with a different expression that evaluates to a tensor of same shape and dtype.
<br>
주어진 파라메터는 어떠한 종류의 심볼릭 변수를 대해하여 사용할수 있지만, shared 변수로는 사용되지 못한다. 
일반적인 변수나 혹은 상수를 대체 할수 있다. 
조심해야 할것은, 그 표현 소개된 순서대로의 의존성이 허용되지 않는다
대체 순서는 정의되지 않는다. , 그래서 대체는 정해지지 않는 순서로 동작한다. 

사실, 좋은 방법 ----
<br>


Using Random Numbers
----------------------
Because in Theano you first express everything symbolically and afterwards compile this expression to get functions, using pseudo-random numbers is not as straightforward as it is in NumPy, though also not too complicated.
The way to think about putting randomness into Theano’s computations is to put random variables in your graph. Theano will allocate a NumPy RandomStream object (a random number generator) for each such variable, and draw from it as necessary. We will call this sort of sequence of random numbers a random stream. Random streams are at their core shared variables, so the observations on shared variables hold here as well. Theanos’s random objects are defined and implemented in RandomStreams and, at a lower level, in RandomStreamsBase.

## Bried Example
--------------------------
	from theano.tensor.shared_randomstreams import RandomStreams
	from theano import function
	srng = RandomStreams(seed=234)
	rv_u = srng.uniform((2,2))
	rv_n = srng.normal((2,2))
	f = function([], rv_u)
	g = function([], rv_n, no_default_updates=True)    	#Not updating rv_n.rng
	nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
	
Here, ‘rv_u’ represents a random stream of 2x2 matrices of draws from a uniform distribution. Likewise, ‘rv_n’ represents a random stream of 2x2 matrices of draws from a normal distribution. The distributions that are implemented are defined in RandomStreams and, at a lower level, in raw_random. They only work on CPU. See Other Implementations for GPU version.
Now let’s use these objects. If we call f(), we get random uniform numbers. The internal state of the random number generator is automatically updated, so we get different random numbers every time.

	f_val0 = f()
	f_val1 = f() # different numbers from f_val0

When we add the extra argument no_default_updates=True to function (as in g), then the random number generator state is not affected by calling the returned function. So, for example, calling g multiple times will return the same numbers.

	g_val0 = g() # different numbers from f_val0 and f_val1
	g_val1 = g()	# same numbers as g_val0!
	
An important remark is that a random variable is drawn at most once during any single function execution. So the nearly_zeros function is guaranteed to return approximately 0 (except for rounding error) even though the rv_u random variable appears three times in the output expression.

	nearly_zeros = function([], rv_u + rv_u -2 * rv_u)
	
##Seeding Streams
Random variables can be seeded individually or collectively.
You can seed just one random variable by seeding or assigning to the .rng attribute, using .rng.set_value().

	rng_val = rv_u.rng.get_valye(borrow=True) # Get the rng for rv_u
	rng_val.seed(89234) # seeds the generator
	rv_u.rng.set_value(rng_val, borrow=True)
	
You can also seed all of the random variables allocated by a RandomStreams object by that object’s seed method. This seed will be used to seed a temporary random number generator, that will in turn generate seeds for each of the random variables.

	srng.seed(902340) # seeds rv_u and rv_n with different seeds each
	
##Sharing Streams Between Functions
As usual for shared variables, the random number generators used for random variables are common between functions. So our nearly_zeros function will update the state of the generators used in function f above.
For example:

	state_after_v0 = rv_u.rng.get_value().get_state()
	nearly_zeros()       # this affects rv_u's generator
	>array([[ 0.,  0.],
       [ 0.,  0.]])
	v1 = f()
	rng = rv_u.rng.get_value(borrow=True)
	rng.set_state(state_after_v0)
	rv_u.rng.set_value(rng, borrow=True)
	v2 = f()             # v2 != v1
	v3 = f()             # v3 == v1
	
##Copying Random State Between Theano Graphs
In some use cases, a user might want to transfer the “state” of all random number generators associated with a given theano graph (e.g. g1, with compiled function f1 below) to a second graph (e.g. g2, with function f2). This might arise for example if you are trying to initialize the state of a model, from the parameters of a pickled version of a previous model. For theano.tensor.shared_randomstreams.RandomStreams and theano.sandbox.rng_mrg.MRG_RandomStreams this can be achieved by copying elements of the state_updates parameter.
Each time a random variable is drawn from a RandomStreams object, a tuple is added to the state_updates list. The first element is a shared variable, which represents the state of the random number generator associated with this particular variable, while the second represents the theano graph corresponding to the random number generation process (i.e. RandomFunction{uniform}.0).
An example of how “random states” can be transferred from one theano function to another is shown below.


	from __future__ import print_function
	import theano
	import numpy
	import theano.tensor as T
	from theano.sandbox.rng_mrg import MRG_RandomStreams
	from theano.tensor.shared_randomstreams import RandomStreams
	
	>>> class Graph():
	...     def __init__(self, seed=123):
	...         self.rng = RandomStreams(seed)
	...         self.y = self.rng.uniform(size=(1,))
	
	>>> g1 = Graph(seed=123)
	>>> f1 = theano.function([], g1.y)
	
	>>> g2 = Graph(seed=987)
	>>> f2 = theano.function([], g2.y)
	
	>>> # By default, the two functions are out of sync.
	>>> f1()
	array([ 0.72803009])
	>>> f2()
	array([ 0.55056769])
	
	>>> def copy_random_state(g1, g2):
	...     if isinstance(g1.rng, MRG_RandomStreams):
	...         g2.rng.rstate = g1.rng.rstate
	...     for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
	...         su2[0].set_value(su1[0].get_values())
	
	>>> # We now copy the state of the theano random number generators.
	copy_random_state(g1, g2)
	f1()
	>array([ 0.59044123])
	f2()
	>array([ 0.59044123])
	
##Other Implementations
There are 2 other implementations based on MRG31k3p and CURAND. The RandomStream only work on the CPU, MRG31k3p work on the CPU and GPU. CURAND only work on the GPU.

`To use you the MRG version easily, you can just change the import to:
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams`

##A Real Example : Logistic Regerssion
	import numpy
	import theano
	import theano.tensor as T
	rng = numpy.random

	N = 400
	feats = 784
	D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
	training_steps = 10000

	# Declare Theano symbolic variables
	x = T.matrix("x")
	y = T.vector("y")
	w = theano.shared(rng.randn(feats), name="w")
	b = theano.shared(0., name="b")
	print("Initial model:")
	print(w.get_value())
	print(b.get_value())

	# Construct Theano expression graph
	p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
	prediction = p_1 > 0.5                    # The prediction thresholded
	xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
	cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
	gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

	# Compile
	train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
	predict = theano.function(inputs=[x], outputs=prediction)

	# Train
	for i in range(training_steps):
   		pred, err = train(D[0], D[1])

	print("Final model:")
	print(w.get_value())
	print(b.get_value())
	print("target values for D:")
	print(D[1])
	print("prediction on D:")
	print(predict(D[0]))