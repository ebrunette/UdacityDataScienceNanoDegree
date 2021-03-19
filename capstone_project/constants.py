model_parameters_dict = {
    'LogisticRegression': {
        'penalty': ['l1','l2'],
        'multi_class': ['auto','ovr']
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini','entropy'],
        'splitter': ['best','random']
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 120, 140, 160, 180, 200],
        'criterion': ['gini', 'entropy']
    },
    'SVC': {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'max_iter': [500,1000]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [.1,.15],
        #'criterion': ['friedman_mse','mse','mae'],
        'loss': ['deviance','exponential']
    }
}

model_names = ['LogisticRegression', 
               'DecisionTreeClassifier', 
               'RandomForestClassifier', 
               'SVC', 
               'GradientBoostingClassifier']

yuan_features = [' Revenue Per Share (Yuan ¥)', 
                 ' Operating Profit Per Share (Yuan ¥)', 
                 ' Per Share Net profit before tax (Yuan ¥)']

log_normal_features = [' Total Asset Turnover',
                       ' Inventory Turnover Rate (times)',
                       ' Cash/Total Assets',
                       ' Long-term Liability to Current Assets',
                       ]