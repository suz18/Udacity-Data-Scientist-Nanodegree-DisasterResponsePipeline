import sys
import pandas as pd
from sqlalchemy import create_engine 


def load_data(messages_filepath, categories_filepath):
    """
    load_data function is to load the messages and categories datasets
    and merge the two datasets.
    
    input: paths for the messages and categories datasets
    
    output: merged dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath) 
    # merge datasets
    df = messages.merge(categories, how = 'inner', on='id')
    return df

def clean_data(df):
    """
    clean_data function is to clean the merged dataset. 
    The function expands category column to 36 individual category columns, renames the columns, 
    encodes column values with 0 and 1, and drops duplicates. 
    
    input: merged dataframe
    
    output: cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0].tolist()
    # use this row to extract a list of new column names for categories.
    category_colnames = [x[:-2] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df.drop_duplicates(inplace = True)
    return df

def save_data(df, database_filepath='data/DisasterResponse.db'):
    """
    save_data function is to save cleaned data into a SQLite database.
    
    input: cleaned dataframe, database path
    
    output: none
    """
    engine = create_engine('sqlite:///./' + database_filepath )
    df.to_sql('messages_clean', engine, index=False) 


def main():
    """
    main function runs all the above functions and produce outputs. 
    """
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
