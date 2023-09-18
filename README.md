AI Based Stock Investing App in Python 

Predictive model for the quantitative analysis of stocks using machine learning / AI (Linear and K-Neighbor Regression Algorithms - Stochastic Gradient Descent (SGD))

(Repo: LinearKNeighborRegressionStockInvestBotWS)

Optimizes a portfolio using stock covariance for volatility minimization and cumulative return maximization with Stochastic Gradient Descent (SGD)  to minimize the negative Sharpe Ratio function for various X parameters representing the different percentage allocations for each Stock.

Note that this minimizes the Sharpe Ratio, not cumulative returns. This is because optimizing for cumulative returns is trivial; merely invest 100% in the one stock that has increased the most! Instead, this minimizes risk as well, which also makes it much more useful for the future since securities tend to maintain the same levels of volatility.

Imagine two companies competing in the same sector. Perhaps one company does better, the other tends to do worse. This is a negative covariance. In this case, it would be possible to attain the returns of both of the stocks with nearly zero risk, as the volatilities of each company become cancelled out if allocations are set evenly to 50% and 50%. This is the big picture of how this optimizes for risk mitigation in addition to cumulative returns.


##### Stochastic Gradient Descent (SGD)##### 

Stochastic Gradient Descent (SGD) is an optimization algorithm often used in machine learning applications to find the model parameters that correspond to the best fit between predicted and actual outputs. It's an inexact but powerful technique. 

While the basic idea behind stochastic approximation can be traced back to the Robbins–Monro algorithm of the 1950s, SGD has become an important optimization method in machine learning. 

SGD is a variant of the Gradient Descent algorithm that is used for optimizing machine learning models. It addresses the computational inefficiency of traditional Gradient Descent methods when dealing with large datasets in machine learning projects.

SGD is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.

SGD is an iterative method often used for machine learning, optimizing the gradient descent during each search once a random weight vector is picked. The gradient descent is a strategy that searches through a large or infinite hypothesis space whenever 1) there are hypotheses continuously being parameterized and 2) the errors are differentiable based on the parameters. The problem with gradient descent is that converging to a local minimum takes extensive time and determining a global minimum is not guaranteed. In SGD, the user initializes the weights and the process updates the weight vector using one data point. The gradient descent continuously updates it incrementally when an error calculation is completed to improve convergence. The method seeks to determine the steepest descent and it reduces the number of iterations and the time taken to search large quantities of data points. Over the recent years, the data sizes have increased immensely such that current processing capabilities are not enough. SGD is being used in neural networks and decreases machine computation time while increasing complexity and performance for large-scale problems. 

Strictly speaking, SGD is merely an optimization technique and does not correspond to a specific family of machine learning models. It is only a way to train a model. Often, an instance of SGDClassifier or SGDRegressor will have an equivalent estimator in the scikit-learn API, potentially using a different optimization technique. For example, using SGDClassifier(loss='log_loss') results in logistic regression, i.e. a model equivalent to LogisticRegression which is fitted via SGD instead of being fitted by one of the other solvers in LogisticRegression. Similarly, SGDRegressor(loss='squared_error', penalty='l2') and Ridge solve the same optimization problem, via different means.


##### Setup##### 

pip install scikit-learn matplotlib pandas scipy


##### Run##### 

Inside the directory you downloaded Investing App, run LinearKNeighborRegressionStockApp.py:

python LinearKNeighborRegressionStockApp.py

The LinearKNeighborRegressionStockApp.py will pull data from Yahoo Finance (can be extended to other APIs) and will walk you through example usages of the Stock and Portfolio classes, like:
- Analyzing and forecasting a single stock as a buy, sell, or holding opportunity
- Scanning all 500 stocks in the S&P 500 index, ranking the top 20 stocks to buy right now
- Optimizing the allocation percentages of stocks in your custom Portfolio to maximize risk-adjusted returns (Sharpe Ratio)
- Maximizing the Sharpe Ratio of all 30 stocks in the Dow Jones Industrial Index by determining optimal percentage allocations

These will also plot charts to display the data. 


##### Example Charts:##### 

## Dow Jones portfolio optimization

## Single-stock analysis and prediction
AAPL 
AMZN 
FB 