import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function: load data from message and categories csv files and merge them
    Args：
      messages_filepath(str): messages file path
      categories_filepath(str): categories files path
    Return：
       df(pd.DataFrame)： merge messages and categories
    """
    #Load the messages csv file
    messages = pd.read_csv(messages_filepath)

    #Load the categories csv file
    categories = pd.read_csv(categories_filepath)

    #merging the two datasets
    df = messages.merge(categories, on='id', how='left')

    return df


def clean_data(df):
    """
    Function: clean input dataframe df by creating one hot encoded category columns and removing duplciates 
    Args：
      df(pd.DataFrame): merged dataframe (from messages and categories)
    Return：
       clean_df(pd.DataFrame)： cleaned dataframe with messages and category columns (1/0)
    """

    #split the categories column into individual category columns
    cat_split = df['categories'].str.split(';', expand=True)

    # rename the columns of `categories`
    cat_split.columns =  cat_split.iloc[0,:].apply(lambda x: str(x)[:-2])

    # one hot encode the category columns
    for col in cat_split.columns:
        # set each value to be the last character of the string  and convert column from string to numeric
        cat_split[col] = cat_split[col].apply(lambda x: str(x)[-1]).astype(int)
        
    clean_df = pd.concat([df.drop('categories', axis=1, inplace = True), cat_split], axis=1)

    # drop duplicates
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)    

    return clean_df    

def save_data(df, database_filename):
    
    """
    Function: save cleaned dataset into a database
    Args:
        df(pd.DataFrame): cleaned dataframe
        database_filename(str): Database name  
    Return:
        NA
    """

    #intialize engine
    engine = create_engine('sqlite://'+database_filename)
    df.to_sql('df', engine, index=False, if_exists = 'replace')


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