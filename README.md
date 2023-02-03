# PSnake-activation-function
An extension to the snake [1] activation function - Snake. Here, every input of the layer gets own trained frequency, instead of only one. The PSnake activation function should be helpful in the situations, where several periodic functions with different frequencies are needed to approximate a given problem. 


# Definition of the function

$$f(x) = x + (1 - cos(2 * alpha * x))/(2 * alpha)$$
where $alpha$ is a learned array with the same shape as $x$ (input).


# References 
Ziyin L, Hartwig T, Ueda M. Neural networks fail to learn periodic functions and how to fix it, arXiv. arXiv preprint arXiv:2006.08195. 2020.

# How to cite the work
Daniel Wiczew, Parametric Snake (PSnake) activation for the neural network to approximate periodic functions. https://github.com/DanielWicz/PSnake-activation-function
