import sys
import re
import pandas as pd
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


def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * FROM messages_df", conn)

    # define features and label arrays
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_colnames = df.columns[-35:]

    return X, Y, category_colnames


def tokenize(text):
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

def build_model():
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
        'clf__estimator__criterion':['gini','entropy'],
        'clf__estimator__min_samples_leaf':[2, 3, 4],
        'clf__estimator__n_estimators': [50, 100, 200]
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters)

    return model_pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Note: Most of the below code was borrowed from the following website 
    https://www.kaggle.com/suveesh/grid-search-pipeline-sample
    '''
    predictions = model.predict(X_test)
    test_model(predictions, Y_test)
    labels = np.unique(predictions)
    confusion_mat = confusion_matrix(Y_test, predictions, labels)
    
    accuracy = (y_pred == y_test).mean()
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)
    pass


def save_model(model, model_filepath):
    pickle.dump(model, open('model.pkl','wb'))
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