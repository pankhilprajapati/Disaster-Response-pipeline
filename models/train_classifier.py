import sys
from sqlalchemy import *
import re
import numpy as np
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib

def load_data(database_filepath):
    '''
    argument:
         database_filepath - path where .db file is stored
    return
         X - feature(message)
         Y - Target
         target_names - target names
    '''

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table_name = database_filepath.split('/')[-1].split('.')[0]
    df = pd.read_sql_table(table_name,engine)
    X = df['message']
    Y = df.iloc[:,4:].values
    target_names = df.iloc[:,4:].columns
    
    return X,Y,target_names

def tokenize(text):
    '''
    Argument:
        text - Raw text
    Return:
        clean_tokens - list of relavant tokens
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
        
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip().lower()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    '''
    Arguments:
        NONE

    Return :
        clf - model is return
    '''
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf',MultiOutputClassifier(OneVsRestClassifier(Perceptron(tol=1e-3, random_state=0))))
            ])
    parameters = {
    'clf__estimator__estimator__alpha':[0.0003, 0.001],
    'clf__estimator__estimator__n_iter_no_change':[5, 10]
    }
    
    clf = GridSearchCV(pipeline, parameters,verbose=3,cv=3)

    return clf


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    Arguments:
        model - model 
        X_test
        Y_test
        category_names - Target Names

    Return :
        None

    '''
    
    y_pred = model.predict(X_test)
    for i in range(y_pred.shape[1]):
        print("Label- {} ".format(category_names[i]))
        print(classification_report(Y_test[:,i], y_pred[:,i],))
        print("")


def save_model(model, model_filepath):
    '''
    Arguments:
        model - model 
        model_filepath - Path where model is store

    Return :
        None

    '''
    joblib.dump(model, model_filepath)


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