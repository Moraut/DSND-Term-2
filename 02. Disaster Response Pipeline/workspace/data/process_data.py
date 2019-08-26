import sys
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    """
    Function: load data from message and categories csv files and merge them
    Args：
      messages_filepath(str): messages file path
      categories_filepath(str): categories files path
    Return：
       df： merge messages and categories
    """
    #Load the messages csv file
    messages = pd.read_csv(messages_filepath + 'messages.csv')

    #Load the categories csv file
    categories = pd.read_csv(categories_filepath + 'categories.csv')

    return messages.merge(categories, on='id', how='left')


def clean_data(df):
    pass


def save_data(df, database_filename):
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()