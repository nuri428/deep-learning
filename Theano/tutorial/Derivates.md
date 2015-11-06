#Dervatives In Theano
##Computing Gradients
Now let’s use Theano for a slightly more sophisticated task: create a function which computes the derivative of some expression y with respect to its parameter x. To do this we will use the macro T.grad. For instance, we can compute the gradient of x^2 with respect to x. Note that: d(x^2)/dx = 2 \cdot x.

그레디언트 계산하기. 

Here is the code to compute this gradient:

	import theano
	import theano.tensor as T
	from theano import pp
	x = T.dscalar('x')
	y = x ** 2
	gy = T.grad(y, x)
	>pp(gy)  # print out the gradient prior to optimization
	'((fill((x ** TensorConstant{2}), TensorConstant{1.0}) * TensorConstant{2}) * (x ** (TensorConstant{2} - TensorConstant{1})))'
	f = theano.function([x], gy)
	f(4)
	>array(8.0)
	f(94.2)
	>array(188.4)


In this example, we can see from pp(gy) that we are computing the correct symbolic gradient. fill((x ** 2), 1.0) means to make a matrix of the same shape as x ** 2 and fill it with 1.0.

'
The optimizer simplifies the symbolic gradient expression. You can see this by digging inside the internal properties of the compiled function.
pp(f.maker.fgraph.outputs[0])
'(2.0 * x)'
After optimization there is only one Apply node left in the graph, which doubles the input.
'

We can also compute the gradient of complex expressions such as the logistic function defined above. It turns out that the derivative of the logistic is: ds(x)/dx = s(x) \cdot (1 - s(x)).

<img src=dlogistic.png >

	x = T.dmatrix('x')
	s = T.sum(1 / ( 1 + T.exp(-x))
	gs = T.grad(s,x)
	dlogistic = theano.function([x], gs)
	dlogistic ( [[0,1], [-1,2]])
	>array([[ 0.25      ,  0.19661193],
       [ 0.19661193,  0.10499359]])
       
 
