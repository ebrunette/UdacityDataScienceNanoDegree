model_parameters_dict = {
    'LinearRegression': {
        'normalize': [True,False]
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

model_names = ['LinearRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'SVC', 'GradientBoostingClassifier']
    