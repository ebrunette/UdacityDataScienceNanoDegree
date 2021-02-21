# import packages
import nltk
nltk.download()
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re

import sys
import pandas as pd 
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
import pickle


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text.strip())
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def load_data():
    # read in file
    messages = pd.read_csv('./messages.csv')
    categories = pd.read_csv('./categories.csv')
    df = categories.merge(messages, on =['id'])
    
    # clean data
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop('categories',axis=1)
    df = pd.concat([df, categories], axis=1)

    df = df.drop_duplicates()
    # load to database
    engine = create_engine('sqlite:///cleaned_message_data.db')
    df.to_sql('messages_df', engine, index=False)

    # define features and label arrays
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)

    return X, Y

def tokenize(text):
    # tokenize text
    tokens = nltk.tokenize.word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens
    
def build_model():
    # text processing and model pipeline
    pipeline_it1 = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = {
        "clf__estimator__criterion":['gini','entropy'],
        "clf__estimator__algorithm":['auto','sqrt','log2'],
        "clf__estimator__n_estimators": [90,100,110,120,130,140,150]
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline_it1, param_grid=parameters)

    return model_pipeline

def test_model(predictions, Y_test):
    """ 
    Prints the output of confusion matrixs for the various columns in the dataset 
    
    input: 
        predictions: The predictions outputted from the model
        Y_test: The array of actual values for the predicitons to test against.
    type_input: 
        preditions: pd.DataFrame
        Y_test: 2D np.array
    
    returns:
        None
    return type: 
        None
    """
    
    for i in range(0, len(predictions[0])):
        Y_test_compare = Y_test.reset_index().drop('index',axis=1)
        print(classification_report(Y_test_compare[Y_test_compare.columns[i]], predictions[:,i]))

def train(X, y, model):
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.33, random_state=42)

    # fit model
    model.fit(X_train, Y_train)

    # output model test results
    predictions = model.predict(X_test)
    test_model(predictions, Y_test)

    return model


def export_model(model):
    # Export model as a pickle file
    pickle.dump(model, open('model.pkl','wb'))


def run_pipeline():
    X, y = load_data()  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    #data_file = sys.argv[1]  # get filename of dataset
    
    run_pipeline()  # run data pipeline