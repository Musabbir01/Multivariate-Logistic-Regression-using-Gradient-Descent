# Multivariate-Logistic-Regression-using-Gradient-Descent
## Gradient Descent
Gradient descent algorithm and its variants ( Adam, SGD etc. ) have become very popular training (optimisation) algorithm in many machine learning applications. Optimisation algorithms can be informally grouped into two categories — gradient-based and gradient-free(ex. particle swarm, genetic algorithm etc.). As you can guess, gradient descent is a gradient-based algorithm. Why gradient is important in training machine learning?
The objective of training a machine learning model is to minimize the loss or error between ground truths and predictions by changing the trainable parameters. And gradient, which is the extension of derivative in multi-dimensional space, tells the direction along which the loss or error is optimally minimized. If you recall from vector calculus class, gradient is defined as the maximum rate of change. Therefore, the formula for gradient descent is simply:
![1_SzinwRriPA6v_USiftpLsQ](https://user-images.githubusercontent.com/54174933/104131366-178e3300-53a0-11eb-98b3-9fdd242303e9.png)

θj is a trainable parameter, j. α is a learning rate. J(θ) is a cost function.
In the below figure, the shortest from the starting point ( the peak) to the optima ( valley) is along the gradient trajectory. The same principle applies the multi-dimensional space which is generally the case for machine learning training.

## Logistic Regression
Logistic Regression was used in the biological sciences in early twentieth century. It was then used in many social science applications. Logistic Regression is used when the dependent variable(target) is categorical.
For example,
To predict whether an email is spam (1) or (0)
Whether the tumor is malignant (1) or not (0)
Consider a scenario where we need to classify whether an email is spam or not. If we use linear regression for this problem, there is a need for setting up a threshold based on which classification can be done. Say if the actual class is malignant, predicted continuous value 0.4 and the threshold value is 0.5, the data point will be classified as not malignant which can lead to serious consequence in real time.
From this example, it can be inferred that linear regression is not suitable for classification problem. Linear regression is unbounded, and this brings logistic regression into picture. Their value strictly ranges from 0 to 1.

![Screenshot (263)](https://user-images.githubusercontent.com/54174933/104131767-c3388280-53a2-11eb-84d5-2e9c8d1a2a81.png)
