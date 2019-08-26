import sys
import pandas as pd
from sqlalchemy import create_engine
import string
import nltk
nltk.download(['punkt','wordnet','averaged_perceptron_tagger'])

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, fbeta_score
from scipy.stats.mstats import gmean

from sklearn.base import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    """
    Function: load cleaned data from database 
    Args：
      database_filepath(str): clean data database filepath
    Return：
       X(pd.DataFrame): Messages dataframe
       Y(pd.DataFrame): Category Flags dataframe 
       category_names: Category column names 
    """
    engine = create_engine('sqllite:///'+ database_filepath)
    df = pd.read_sql('df', engine)

    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns

    return X,Y,category_names 

def tokenize(text):
    """"
    Function: tokenize text
    Args:
        text(str): Input text"
    Return:
        tokens(list): tokenized text
    """
    remove_punc_table = str.maketrans('', '', string.punctuation)
    stop_words = nltk.corpus.stopwords.words("english")
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    #normalize
    text = text.translate(remove_punc_table).lower()

    #tokenize text
    tok_text = nltk.word_tokenize(text)

    tokens = [lemmatizer.lemmatize(x) for x in tok_text if x not in stop_words]

    return tokens

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


def build_model():
    """
    Function: Build Model function
    
    This function outputs  a Scikit ML Pipeline that processes 
    text messages as per NLP best-practice and 
    applies a classifier.
    
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('first_verb', first_verb())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline


def multioutput_fscore(y_true, y_pred, beta=1):
    """
    Function: MultiOutput Fscore
    
    This is a custom performance metric due to the multilabel outputs.
    It is a sort of geometric mean of the fbeta_score, computed on each label.
    
        
    It can be used as scorer for GridSearchCV:
        scorer = make_scorer(multioutput_fscore,beta=1)
        
    Args:
        y_true: labels
        y_pred: predictions
        beta: beta value of fscore metric
    
    Return:
        f1score -> customized fscore
    
    """
    scores = []

    if isinstance(y_pred, pd.DatFrame) == True:
        y_pred = y_pred.values
    
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values

    
    for col in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,col], y_pred[:,col], beta, average='weighted')
        scores.append(score)

    f1score_np = np.array(score_list)
    f1score_np = f1score_np[f1score_np<1]

    # Geometric Mean of all the f1score to get a weighted estimate of the overall performance    
    f1score = gmean(f1score_np)

    return f1score


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function: Evaluate Model Function

    This function applies the ML pipeline to test set and prints out 
    the model performance.

    Args:
        model(scikit pipeline): Model Pipeline
        X_test(pd.DataFrame): Test Features
        Y_test(pd>DataFrame): Test Messages categories
    Return:
        Prints out model performance:  F1_s 

    """
    # Generate predictions on test data
    y_pred = model.predict(X_test)

    multi_f1 = multioutput_fscore(Y_test, y_pred)
    overall_accuracy = (Y_test == y_pred).mean().mean()

    print("Overall acuracy of the model is {0.2f}%\n".format(overall_accuracy*100))
    print("Custom f1_score of the model is {0.2f}%\n".format(multi_f1*100))

    # Generate category wise performance
    y_pred_df =  pd.DataFrame(y_pred, columns = Y_test.columns)

    for col in y_pred_df.columns:
        f_score = fbeta_score(Y_test[col],y_pred_df[col],, beta=1.0, average='weighted')
        accuracy = (Y_test[col],y_pred_df[col]).mean()
        print("Accuracy for Category || {} :{0.2f%}".format(col, accuracy*100)
        print("F1 Score for Category || {} :{0.2f%}".format(col, f_score*100))


def save_model(model, model_filepath):
    """"
    Function: Save the model as pickle file

    Args:
        model(scikit pipeline): model pipeline
        model_filepath(str): file name for saved model   

    Return:
        NA

    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    

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