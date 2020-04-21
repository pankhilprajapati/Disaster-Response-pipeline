import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import *
from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import STOPWORDS


app = Flask(__name__)

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
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():


    '''
    first is the pie chart displays the Classification of the 
    message 

    '''
    # this the data wrangle for the PIE CHART
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    target = df.iloc[:,4:]
    target_c = target.sum().sort_values(ascending=False)
    pieChart = target_c.to_frame().reset_index()
    labels = pieChart['index'].values.tolist()
    values = target_c.values.tolist()



    ### This is the Wrangling for the bar charts
    # this function generates the n grams of the text passed
    # into it
    def gen_ngrams(text, n_gram=1):
        '''
          this function generates the n grams of the text passed
          into it

          Arguments:
                 text - take raw text 
                 n_gram - no of grams 1,2,3.. .
          return:
                 list of the ngrams 
                 
        '''
        token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
        ngrams = zip(*[token[i:] for i in range(n_gram)])
        return [' '.join(ngram) for ngram in ngrams]
    
    N=100

    message_unigrams = defaultdict(int)
    for tweet in df['message']:
        for word in gen_ngrams(tweet):
            message_unigrams[word] += 1
    
    df_message_unigrams = pd.DataFrame(
                        sorted(message_unigrams.items(),
                        key=lambda x: x[1])[::-1]
                        )
    # Bigrams
    message_bigrams = defaultdict(int)
    for tweet in df['message']:
        for word in gen_ngrams(tweet, n_gram=2):
            message_bigrams[word] += 1
    
    df_message_bigrams = pd.DataFrame(
                        sorted(message_bigrams.items(), 
                        key=lambda x: x[1])[::-1]
                        )

    # Trigrams
    message_trigrams = defaultdict(int)


    for tweet in df['message']:
        for word in gen_ngrams(tweet, n_gram=3):
            message_trigrams[word] += 1
    
    df_message_trigrams = pd.DataFrame(
                        sorted(message_trigrams.items(), 
                        key=lambda x: x[1])[::-1]
                        )



    graphs = [
        {
            'data': [
                Pie(
                    labels=labels, 
                    values=values
                )
            ],

            'layout': {
                'title': 'Categories of Messages',
            }
        },{
            'data':[
                Bar(
            x=df_message_unigrams[1].values[:N],
            y=df_message_unigrams[0].values[:N],
            orientation='h',
          marker=dict(
        color='rgba(255, 0, 0, 0.6)'
    )
      )
            ],
            'layout': {
                'title': 'Top most common Unigrams in messages',
                'yaxis': {
                    'title': "Unigrams"
                },
                'xaxis': {
                    'title': "Counts"
                }
            }
        },{
            'data':[
                Bar(
            x=df_message_bigrams[1].values[:N],
            y=df_message_bigrams[0].values[:N],
            orientation='h',
          marker=dict(
        color='rgba(0, 0, 255, 0.6)'
    )
      )
            ],
            'layout': {
                'title': 'Top most common Bigrams in messages',
                'yaxis': {
                    'title': "Bigrams"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        },{
            'data':[
                Bar(
            x=df_message_trigrams[1].values[:N],
            y=df_message_trigrams[0].values[:N],
            orientation='h',
          marker=dict(
        color='rgba(0, 255, 0, 0.6)'
    )
      )
            ],
            'layout': {
                'width':1000,
                 'height':500,
                'title': 'Top most common Trigrams in messages',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Trigrams"
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



if __name__ == '__main__':
    main()