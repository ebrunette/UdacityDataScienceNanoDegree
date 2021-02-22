import sys
import pandas as pd 
from sqlalchemy import create_engine
import sqlite3
import os

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges the data. 

    input: 
        messages_filepath: The filepath for loading in the messages data 
        categories_filepath: The filepath for leading in the messages data
    type_input: 
        messages_filepath: str
        categories_filepath: str
    output: the merged dataframe of the input files
    type_output: pd.DataFrame
    returns: A raw merged dataframe based on passed in file paths.  
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = categories.merge(messages, on =['id'])
    return df


def clean_data(df):
    """
    Cleans the dataframe by cleaning up the coclumns, and splitting up some of the data to make it a tidy dataset. 

    input: The dataframe from the load_data method. 
    type_input: pd.DataFrame
    output: A modified dataframe with the changes mentioned. 
    type_output: pd.DataFrame
    returns: A dataframe near ready for pipeline for modeling and exporting to database. 
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df = df.drop('categories',axis=1)
    df = pd.concat([df, categories], axis=1)

    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Outputs the dataframe to a database with the passed in database name. 
    If the database exists, then this method will delete the old database before saving it again. 

    input: 
        df: Dataframe representing cleaned data from the original inputs
        datbase_filename: The string of the new database to be created. 
    type_input: 
        df: pd.DataFrame
        database_filename: str
    output: 
        A db file in the same directory as this file. 
    type_output: 
        database_filename.db file
    returns: Nothing to the main process but outputs the db file with the appropriate database_filename
    """
    try: 
        engine = create_engine(database_filename)
        df.to_sql('messages_df', engine, index=False)
    except: 
        print("Deleting old database.")
        os.remove('./data/{}'.format('DisasterResponse.db'))
        
        print("Creating new database") 
        engine = create_engine(database_filename)
        df.to_sql('messages_df', engine, index=False)
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