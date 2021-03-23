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
                       ' Interest-bearing debt interest rate',
                       ' Tax rate (A)',
                       ' Net Value Per Share (B)',
                       ' Net Value Per Share (A)',
                       ' Net Value Per Share (C)',
                       ' Persistent EPS in the Last Four Seasons',
                       ' Cash Flow Per Share',
                       ' Revenue Per Share (Yuan ¥)',
                       ' Operating Profit Per Share (Yuan ¥)',
                       ' Per Share Net profit before tax (Yuan ¥)',
                       ' Realized Sales Gross Profit Growth Rate',
                       ' Continuous Net Profit Growth Rate', 
                       ' Net Value Growth Rate',
                       ' Total Asset Return Growth Rate Ratio',
                       ' Cash Reinvestment %',
                       ' Current Ratio',
                       ' Quick Ratio',
                       ' Total debt/Total net worth',
                       ' Debt ratio %',
                       ' Long-term fund suitability ratio (A)',
                       ' Borrowing dependency',
                       ' Contingent liabilities/Net worth', 
                       ' Operating profit/Paid-in capital',
                       ' Net profit before tax/Paid-in capital', 
                       ' Inventory and accounts receivable/Net value',
                       ' Total Asset Turnover',
                       ' Accounts Receivable Turnover', 
                       ' Average Collection Days', 
                       ' Net Worth Turnover Rate (times)', 
                       ' Revenue per person', 
                       ' Operating profit per person', 
                       ' Allocation rate per person',
                       ' Cash/Total Assets', 
                       ' Quick Assets/Current Liability', 
                       ' Cash/Current Liability', 
                       ' Current Liability to Assets', 
                       ' Operating Funds to Liability',
                       ' Inventory/Working Capital',
                       ' Current Liabilities/Equity',
                       ' Long-term Liability to Current Assets',
                       ' Total income/Total expense',
                       ' Total expense/Assets', 
                       ' Fixed Assets to Assets',
                       ' Current Liability to Equity',
                       ' Equity to Long-term Liability',
                       ' Liability-Assets Flag',
                       ' Total assets to GNP price', 
                       ' Liability to Equity',
                       ' Degree of Financial Leverage (DFL)', 
                       ' Equity to Liability']