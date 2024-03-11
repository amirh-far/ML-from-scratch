# Logistic Regression

This model is kinda different from the linear regression because it is used to classify instead of predict a value. Also, the form of the line is from a sigmoid function.  
This model is used to predict and estimate the probability that an instance belongs to a particular class.  
If exceeding a probability threshold, which is usualy 50%, then model predicts that the instance belongs to a class.  

## Hypothesis Function

The hypothesis function is:  
$`\hat{p} = h_{\theta}(x) = sigmoid(\theta^{T}x) `$  
The $`\theta`$ vector(parameter vector) is a vertical vector.  
Now let's talk about the linear behavior.  

## Linear behavior

The linear behavior of logistic regression is a sigmoid function which outputs a number between 0 and 1.  
The product of the sigmoid function is a number between 0 to 1. if the output is higher than 0.5 then the prediction is 1 and if lower, the prediction is 0.  

Note: You can guess that the behavior of the sigmoid shows that if the $`t`$ in the $`sigmoid(t)`$ is > 0, then the output is higher than 0.5 leading to 1 in the end.  
And if $`t`$ is < 0 then the output is lower than 0.5 leading to 0 in the end.  

## How does the algorithm work?

No closed-form equation to compute the value of $`\theta`$ is available. How ever, since the cost function is convex, the gradient descent algorithm (or other optimization algorithm) is guaranteed to find the global minimum.  
partial derivative of logistic function(to use gradient descent) w.r.t each parameter:

$` \frac{\partial J(\theta)}{\partial \theta_{j}} =
 \frac{1}{m} \sum_{i = 1}^{m} (sigmoid(\theta^{T}x^{(i)}) - y^{(i)}) x_{j}^{(i)}`$  

Now, the way we solve the problem is the same as batch gradient descent(you can use stochastic GD or mini-batch GD) for linear regression.  

If we remove the sum and simplify the eqation, we have:  
$`
\frac{\partial J(\theta)}{\partial \theta_{j}} =
\frac{1}{m} X^{T}(sigmoid(X\theta) - y)
`$  
We use this eqation to find the gradients and therefore find the parameter vector.  