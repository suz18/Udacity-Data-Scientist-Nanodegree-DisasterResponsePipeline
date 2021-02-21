import sys
import pandas as pd
from sqlalchemy import create_engine 

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath):
    """
    load_data function loads data from the SQLite database.
    
    input: database path
    
    output: X, Y and category names that are to be fed in the model. 
    """
    engine = create_engine('sqlite:///./' + database_filepath)
    df = pd.read_sql_table('messages_clean', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """
    The function tokenizes text. 
    
    input: text 
    
    output: clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    This function builds a text processing and machine learning pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__max_df': (0.90, 1.0),
        'vect__min_df': (0.10, 0.20),
        'tfidf__smooth_idf': (True, False), 
        'clf__estimator__min_samples_leaf': (1, 2),
        'clf__estimator__min_samples_split': (2, 3)
        
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    The function exvaluates the model and outputs results on the test set. 
    
    input: model, X_test, Y_test, category_names
    
    output: results on the test set
    """
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print('F1 score, precision and recall for category "', category_names[i], '" is shown as below: ')
        print(classification_report(Y_test.iloc[:, i:(i+1)], Y_pred[:, i:(i+1)]))


def save_model(model, model_filepath):
    """
    This function exports the final model as a pickle file. 
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Mian function runs all the above functions and generates outputs. 
    """
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
