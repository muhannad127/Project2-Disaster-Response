from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
from joblib import dump, load
import sqlite3
import numpy as np
import pandas as pd
import sklearn
import re
import sys
import nltk


nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):

    """
    DESCRIPTION: 
                load dataset from SQL database and split to X and Y
                to train model
    
    INPUT: 
            database_filepath: (str) path of database 
        
    OUTPUT:
        X: (pandas.Series) messages 
        Y: (pandas.DataFrame) Categories
        Y.columns: (list) a list of the category names
    """
    # create engine for SQL database
    engine = create_engine('sqlite:///'+database_filepath)
    
    # Load 'mescat' table to df
    df = pd.read_sql_table('mescat', engine)
    # Get only message col and put in X
    X = df['message'] 
    # Put category vals into Y
    Y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)
    return X, Y, Y.columns
    

def tokenize(text):

    """
    DESCRIPTION: 
                Tokenize, lemmetize, and normalize text before
                feeding to machine learning model
    
    INPUT: 
            text: (str) message to tokenize 
        
    OUTPUT:
        clean_tokens: (list) list of processed tokens 
    """
    
    #tokenize text
    tokens = word_tokenize(text)
    # Instanciate lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    
    #iterate through tokens and lemmitize and normalize
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    
def build_model():

    """
    DESCRIPTION: 
                Build SVC model with optimal params found using GridSearch
    
    INPUT: 
            None
        
    OUTPUT:
        cv: (sklearn.model_selection.GridSearchCV) model 
    """
    
    # the ml pipeline
    # use 'liblinear' to test both l1 and l2 
    pipeline = Pipeline([
    ('vec', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ( 'multi_out_clf', MultiOutputClassifier(LogisticRegression(solver='liblinear')))
        
    ])
    
    # paramaters to try and choose from
    parameters = {
        'vec__ngram_range': ((1, 1), (1, 2)),
        'vec__max_df': (0.75, 1.0),
        'vec__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'multi_out_clf__estimator__penalty': ['l1', 'l2']
    }
    
    # Instanciate GridSearchCV object to find best params and select best model
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """
    DESCRIPTION: 
                Output metrics to analyze model performance
    
    INPUT: 
            model: (sklearn.pipeline.Pipeline) model
            X_test: (numpy array) test data features
            y_test: (pandas.DataFrame) test data True out
            category_names: (list) y cols
        
    OUTPUT:
        None
    """
    # predict on x_test
    Y_pred= model.predict(X_test)
    # convert Y_test to numpy array
    Y_test_np= Y_test.to_numpy()
    
    # iterate through every col to compute metrics
    for i in range(Y_pred.shape[1]):
        print('CATEGORY:', category_names[i])
        print(classification_report(Y_test_np[:,i], Y_pred[:, i]))
        print('\n\n')
    


def save_model(model, model_filepath):
    
    """
    DESCRIPTION: 
                Output metrics to analyze model performance
    
    INPUT: 
            model: (sklearn.pipeline.Pipeline) model
            model_filepath: (str) where to save model
        
    OUTPUT:
        None
    """
    
    #dump model
    dump(model, model_filepath) 
    
    

def main():
    
    ''' Main function to train, evaluate, and save machine learning model'''
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
