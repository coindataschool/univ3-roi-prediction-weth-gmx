# Machine Learning Predictions of ROI and Fee APR of Univ3 WETH-GMX positions on Arbitrum

Using historical Univ3 pool positions data obtained from Revert Finance, we built
XGBOOST regression models to predict ROI and Fee APR using fee tier, 
lower limit price, upper limit price, price range, and duration.

The dashboard compares the predictions with the actuals on a set of data that weren't
used at all during model training and selection via cross-validation. It also 
allows users to enter their own numbers and get back model predicted results. 

![screen]()

The models are limited by the data used for training. Use with caution.

[Dashboard Link]()
