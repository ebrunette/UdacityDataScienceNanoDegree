import sys
import re
import pandas as pd
import numpy as np
import sqlite3
import nltk
nltk.download('averaged_perceptron_tagger')
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle
import os
from pathlib import Path

def load_data(database_filepath):
    """
    Loads in the data from a database for training. 

    input: A string passed in from the command line for running the data. 
    type_input: str
    output: 
        X: The feature data used for training the model. 
        Y: The features that the model will be predicting
        category_colnames: The predicted column names
    type_output: 
        X: pd.Dataframe
        Y: pd.DataFrame
        category_colnames: list
    returns: The data that is required for training the model. 
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM messages_df", conn)

    # define features and label arrays
    # if the 'related' column is a 2, are primarily composed of tweets that are in a different language. 
    # I think it would be best to remove them from the dataset to train on. 
    # As foreign languages are likely a subset and we probably don't have enough information to train an accurate model. 
    df = df[df['related'] != 2]
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_colnames = df.columns[-36:]

    return X, Y, category_colnames


def tokenize(text):
    """
    Cleans the text and prepares it for modeling. 

    input: String that represents the tweet text. 
    type_input: str
    output: The cleaned tokens used for modeling
    type_output: List
    Returns: A list of elements tokenized for modeling. 
    """
    # tokenize text
    tokens = nltk.tokenize.word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Internal class used for extracting the starting verb for another feature for modeling

    Input: 
        BaseEstimator: Base estimator class from sklearn.base
        TransformerMixin: Base estimator class from sklearn.base
    Type_intput: 
        BaseEstimator: BaseEstimator 
        TransformerMixin: TransformerMixin
    Output: N/A
    Type_output: None
    Returns: An object used for providing transformation method for the pipeline of training the model.
    """
    def starting_verb(self, text):
        """
        Method does part of speech tagging for text passed in and returns if it is a starting verb in the string representing a tweet. 

        Input: 
            text: The text of the tweet for disaster recovery. 
        Type_input: 
            text: str
        Output: 
            True: If there is a verb in the sentence 
            False: If there isn't a verb in the sentence. 
        type_output: Bool
        Returns: A boolean to determine if there is a starting verb in the text for classification. 
        """
        sentence_list = nltk.sent_tokenize(text.strip())
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        """
        A stubbed out method for fitting a model that doesn't apply here but allows this class to be used in a pipeline. 
        No more documentation will be filled out until this method get's filled in. 
        """
        return self

    def transform(self, X):
        """
        Method that transfroms the incoming text data to determine if there is a starting verb. 

        input: 
            X: A list representing the tweet data 
        type_input: 
            X: list
        output: 
            X_tagged: DataFrame with the transformed data representing a new feature for modeling. 
        type_output: 
            X_tagged: pd.DataFrame
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Builds the pipeline and Grid Search CV used for training the model

    input: N/A
    type_input: None
    output: The model pipeline as a GridSearchCV object
    type_output: GridSearchCV
    Returns: The pipelined GridSearchCV object used in training and predicting the model. 
    """
    # text processing and model pipeline
    pipeline = Pipeline([
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
        #'clf__estimator__criterion':['gini','entropy'],
        'clf__estimator__min_samples_leaf':[2],
        #'clf__estimator__n_estimators': [50, 100, 200]
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return model_pipeline

def test_model_outcome(predictions, Y_test, category_names):
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
    print(category_names)
    print(len(predictions[0]))
    
    for i in range(0, len(predictions[0])):
        print("Category Name: {}".format(category_names[i]))
        Y_test_compare = Y_test.reset_index().drop('index',axis=1)
        print(classification_report(Y_test_compare[Y_test_compare.columns[i]], predictions[:,i]))

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model using the category column names. 

    Input: 
        model: A trained model that is the subject to evaluation.
        X_test: The dataframe representing the test data for making predictions to test
        Y_test: The dataframe representing the actual categorization for making predictions against
        category_names: The column names representing the predicted features 
    Type_input: 
        model: Grid Search CV object
        X_test: pd.DataFrame
        Y_test: pd.DataFrame
        category_names: list
    Output: Output's the results to consol
    type_output: N/A
    Returns: The evaluation metric results for the model. 
    """
    predictions = model.predict(X_test)
    test_model_outcome(predictions, Y_test, category_names)
    pass


def save_model(model, model_filepath):
    """
    Outputs the model to a pickle object for reference later in the project. 

    inputs: 
        model: A trained model for output. 
    type_inputs: 
        model: GridSearchCV
    output: N/A
    type_output: None
    Returns: Doesn't return anything, but does save a pkl file in the directory for loading in the web app. 
    """
    output_path = os.path.join(os.getcwd(), model_filepath)

    pickle.dump(model, open(output_path,'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)



        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()