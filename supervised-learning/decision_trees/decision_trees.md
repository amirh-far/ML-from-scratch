# Decision Trees

Decision trees are kind of versatile ml algorithms that can perform both classification and regression tasks, and even multioutput tasks. They are capable of fitting complex datasaets.  

The good thing about decision trees is that they dont require much data preparation.For example they dont need feature scaling or centering at all.  

I will implement the decision tree using CART algorithm. This algorithm will produce binary tress. On the other hand, There are other algorithms like ID3 which is used for classification task only with nodes that have more than two children.  

CART algorithm stands for Classification and Regression Tree.  

There are 2 popular ways to find the impurity of a class: Gini Impurity and Entropy(with Information Gain).  
We will use the Entropy and then calculate the Information Gain of each class.  
