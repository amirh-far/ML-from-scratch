# Linear Regression

linear regression model prediction formula:

$ y = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + ... + \theta_{n}x_{n}   $  
where $\theta_{0}$ is the bias term and other thetas are the feature weights.  
  
which is equal to:  
$ y = h_{\theta}(x) = \theta . x$  
  
$ h_{\theta}(x) $ is the hypothesis function  
$ \theta $ is the parameter vector  

## Parameters

### RMSE & MSE

To train the model aka set the parameters, we need to measure how well the model fits the training data (we need a cost function).  
The most common performance measure for regression models is RMSE or root mean square error.  

It is simpler to use MSE (mean squared error) instead of the RMSE and it leads to the same result.  

MSE formula:

$ MSE =(X, h_{\theta} = \frac{1}{m}\sum_{i = 1}^{m} (\theta^{T}x^{(i)} - y^{(i)})^{2} $  

so we find the value of $\theta$ (the parameter vector) so that it minimizes the MSE. how?  
With Normal Equation.  

### Normal Equation

To find the value of $\theta$ that minimizes the MSE, we can get exactly that with Normal Equation.
