# Linear Regression

linear regression model prediction formula:

$` y = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + ... + \theta_{n}x_{n}   `$  
where $\theta_{0}$ is the bias term and other thetas are the feature weights.  
  
which is equal to:  
$` y = h_{\theta}(x) = \theta . x `$  
  
$` h_{\theta}(x) `$ is the hypothesis function  
$` \theta `$ is the parameter vector  

## Parameters

### RMSE & MSE

To train the model aka set the parameters, we need to measure how well the model fits the training data (we need a cost function).  
The most common performance measure for regression models is RMSE or root mean square error.  

It is simpler to use MSE (mean squared error) instead of the RMSE and it leads to the same result.  

MSE formula:

$` MSE =(X, h_{\theta}) = \frac{1}{m}\sum_{i = 1}^{m} (\theta^{T}x^{(i)} - y^{(i)})^{2} `$  

so we find the value of $`\theta `$ (the parameter vector) so that it minimizes the MSE. how?  
With Normal Equation.  

### Normal Equation

To find the value of $` \theta `$ that minimizes the MSE, we can get exactly that with Normal Equation.  

Normal Equation:  
$`\theta = (X^{T}X)^{-1} X^{T} y `$

Via this equation, we can now have the best theta values.  
Note that we can use Gradient Descent to enhance this regression.

> Quick math note:
The inverse of $`A`$ is $`A^{-1}`$ only when:  
$` AA^{-1} = A^{-1}A = I `$

### Some notes

In order to perform linear regression, we need to add a dummy feature aka intercept term. As we mentioned in the formula above, the bias term is $`\theta_{0}`$ which has not multiplied by any column of x. If we add a dummy column 1 as intercept term, then we would have $`x_{0}`$ equal to 1 and then any number multiplied by 1 is equal to the number itself which is going to be $`\theta_{0}`$.

#### How the algorithm is applied

So what we do in the simplest form of linear regression without any regularization, we have to choose the cost function; which we use MSE. Then, in order to calculate the parameter vector (in our case the $`\theta`$ vector) with this cost function, we use the Normal Equation which is: $`\theta = (X^{T}X)^{-1} X^{T} y `$. This function, minimizes the Mean Squared Error.

## Batch Gradient Descent

Previously we used normal equation to calculate the parameter vector. This approach has its own ups and downs. This approach is limited and its usage is mostly on small datasets.  
The more frequent approach is using Gradient Descent because this approach does better in large datasets and feature scaling can be performed in this approach. For more advantages & disadvantages check the refrences section.  

### How does Gradient Descent work?

This algorithm, calculates the Gradient of the loss function and moves towards the descending of the loss function. In other words, it will calculate the minimum of the loss function. As the name applies, the Gradient is the partial derivative of the loss function w.r.t to each parameter.  
We use the MSE this time and doing a partial derivative from MSE w.r.t the parameter vector is equal to:  
$`\frac{\partial MSE}{\partial \theta} = X^{T}(X\theta - y) `$

## Refrences

- [Hands on Machine Learning Book](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1098125975)
- [100 page ML book](https://www.amazon.com/Hundred-Page-Machine-Learning-Book/dp/199957950X)
- [Matrix inversion](https://www.mathsisfun.com/algebra/matrix-inverse.html)
- [Matrix Transpose](https://mathinsight.org/matrix_transpose)
- [Gradient Descent vs. Normal Equation](https://www.geeksforgeeks.org/difference-between-gradient-descent-and-normal-equation/)
- [Gradient Descent, Step-by-Step video(recommended!)](https://www.youtube.com/watch?v=sDv4f4s2SB8)
