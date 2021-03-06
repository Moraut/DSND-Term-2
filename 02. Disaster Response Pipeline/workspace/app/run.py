import json
import plotly
import pandas as pd

import nltk
nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class first_verb(BaseEstimator, TransformerMixin):
    
    def first_verb(self, message):
        sent_list = nltk.sent_tokenize(message)
        for x in sent_list:
            pos_tags = nltk.pos_tag(tokenize(x))
            if len(pos_tags) >0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            else:
                False
        return False
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X):
        X_tag = pd.Series(X).apply(self.first_verb)
        return pd.DataFrame(X_tag)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visual: # of related vs not-realted messages
    related_dict = {0:'not realted', 1:'related'}

    rel_df = df['related'].map(related_dict)
    related_counts = rel_df.value_counts()
    related_names  = list(related_counts.index)

    # create visual: top 10 categories percentage distribution of messages (excluding related)

    categories = df.iloc[:,4:]
    categories_mean = categories.mean().sort_values(ascending=False) [1:11]
    categories_names = list(categories_mean.index)


    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        #Visual 1
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Messages by Genres',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis':{
                    'title': "Genre"
                }
            }
        },

        #Visual 1
        {       
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Related vs Non Related',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related/Non Related"
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