# PSnake Activation Function - A Powerful Tool for Periodic Data Approximation

PSnake is an extension to the snake activation function that enables every input of the layer to have its own trained frequency, instead of only one. This activation function is particularly useful in situations where several periodic functions with different frequencies are needed to approximate a given problem, such as in finance, stock, economical data, weather prediction, sound, EKG signal, and other fields.

The PSnake activation function forms some sort of basis function for the trained periodic signal, and can be used as a powerful tool for approximating periodic data. The function definition is given as follows:

# Example of applications
The PSnake activation function can also be applied to fields such as:

 * Signal Processing
 * Time-series data
 * Predictive maintenance
 * Speech recognition
 * Music analysis
 * Image processing

# Activation function definition

The PSnake activation function is defined as:

$$f(x) = x + (1 - cos(2 * \alpha * x))/(2 * \alpha)$$

where $\alpha$ is a learned array with the same shape as the input $x$.


# References 
Ziyin L, Hartwig T, Ueda M. Neural networks fail to learn periodic functions and how to fix it, arXiv. arXiv preprint arXiv:2006.08195. 2020.

#How to Cite

If you find this work useful in your research, please consider citing:
Daniel Wiczew, Parametric Snake (PSnake) activation for the neural network to approximate periodic functions. GitHub repository
Cite also original work from Ziyin L. et al.
