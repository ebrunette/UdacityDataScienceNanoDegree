import json
import plotly
import pandas as pd
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter

from flask import Flask
from flask import render_template, request, jsonify, redirect, url_for, Response
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

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

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_df', engine)

# Data for visuals
EDA_df = pd.DataFrame()
EDA_df['message_length'] = df['message'].apply(lambda x: len(x))
EDA_df['id'] = df['id']
text = df['message'].values

counts,bins = np.histogram(EDA_df.message_length, bins=range(0,1000, 100))
bins = 0.5 * (bins[:-1] + bins[1:])

stop = stopwords.words('english')
punctuation_to_remove = ['-',',','.','..','(',')']
for punct in punctuation_to_remove:
    stop.append(punct)
EDA_df['tweet_without_stopwords'] = df['message'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in (stop)]))
most_common = Counter(" ".join(EDA_df['tweet_without_stopwords']).split()).most_common(20)
to_add = {}
to_add['words'] = []
to_add['counts'] = []
for count in most_common:
    to_add['words'].append(count[0])
    to_add['counts'].append(count[1])
most_popular = pd.DataFrame(to_add)

# load model
model = joblib.load("../model/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
#@app.route('/plot.png')

def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=bins, y=counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Lengths (Less than 1000 characters)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Length"
                }
            }
        },
        {
            'data': [
                Bar(x=most_popular['words'], y=most_popular['counts'])
            ],
            'layout': {
                'title': 'Top 20 words in the training set.',
                'yaxis': {
                    'title': "Words"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()