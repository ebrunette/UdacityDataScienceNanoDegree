# Project Motivation
Usually personal stories don't come in READMEs, but this one can be a little different. My dad owned a small business that went bankrupt when I was a kid. He had some signs that he was going under, but I don't know if he recognized them, particularly not the financial ones. With this project, I am hoping to find the factors that can help other small businesses to find some warning signs in their businesses, and can help later on with doing some changes of how their business is structured or start a new change in business operations to increase the factors that help a sucessful business. 

# Problem Statement
Is there any features from this dataset that will help predict if a company is going to go under, and if there is one that stands out, why? What model produces the best results, and why might that model be the best?  

# Metrics used to determine this? 
A confusion metric is the best result for the analysis 

# Required Libraries 
1. pandas
2. matplotlib
3. sklearn
4. pickle
5. seaborn

# File Structure
* Bankrupt_EDA.ipynb
* constants.py
* models
    * Models that are saved via pickle package. 
* data
    * data.csv

# File/Folder explanations
- Bankrupt_EDA.ipynb
    1. This file is the main notebook for exploring, training, and evaluating the models for this project. 
- constants.py
    1. This file is to help extract data that might clutter the main notebook with information that can be extracted and isn't applicable for most of the eda analysis. 
- models
    1. This is a folder that is created to keep the model outputs stored somewhere and are required for the training aspect becuase some models take a while to train.
    2. The trained models are a pain to get into Git because of the size of the output Pickle files, so this folder is here to hold the models that are created. 
- data/data.csv
    1. This file is the data that was gathered from the [Kaggle Dataset](https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction)

# Project Origin and Related Datasets
The data was gathered through this [Kaggle dataset](https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction). The dataset originated from the Taiwan Econmic Journal from the years 1999 to 2009. The data was obtained from [UCI Machine Learning](https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction) library. As seen in their documentation for the Kaggle dataset. 

# Model Selection 
1. [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) 
2. [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
3. [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
4. [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
5. [XGBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

# Model Refinement
The base iteration didn't have any changes to the dataset. The data was all double values and was a great dataset for jumping in and doing some modeling. 
The first iteration was turning only features containing Yuan (the currency for Taiwan) from a logistic distribution to a normal distribution for modeling. 
The second iteration turned a large number of features into logistic distribution to a normal distribution like in the second iteration. 

# Conclusions
The model that performed the best in all three iteration was Decision Tree. The features that were the best predicters for banruptcy are Borrowing dependency, Net Value Growth Rate, and Non-industry income and expenditure/revenue based on the dataset. 

# Future Work
The future work might be adjusting the features a little more by removing some of the features that I picked that might not have had a lognormal distributionl. This would hopefully cause the feature weights to be closer together.  

# Necessary Acknowledgements
I would like to thank the graders Udacity for helping me iterating on the data and raising this project to their standards, and helping me learn the information to make this project possible. 
I would also like to thank Kaggle for providing this dataset. 