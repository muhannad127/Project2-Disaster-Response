import numpy as np
import pandas as pd
import sklearn
import sqlite3
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    DESCRIPTION: 
                Load two csv files into a dataframe, merge them,
                and return the result  

      INPUT: 
            messages_filepath: (str) path of messages data
            categories_filepath: (str) path of categories data
        
     OUTPUT:
            df: (pandas.DataFrame) dataframe of merged messages and categories data
    """
    
    #messages
    messages = pd.read_csv(messages_filepath) 
    #print(messages.head())
    
    #their categories
    categories = pd.read_csv(categories_filepath) 
    
    #merged on shared 'id'
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):

    """
    DESCRIPTION:
            Get data ready for ml model: split and replace category vals with
            numeric values indicating status of category, remove duplicates,
            and rename category columns  

    INPUT: 
        df: (pandas.dataframe) data to be cleaned 
    
    OUTPUT:
        df_clean: (pandas.DataFrame) clean data
    """
    # Split categories to different columns
    #print(df.columns)
    categories = df['categories'].str.split(pat=';',expand=True)
    
    # Get Category names by accessing first row
    row = categories.loc[0].to_list()
    
    # Define func to drop last two chars of str 
    #eg 'related-0' -> 'related'
    func= lambda string: string[:-2] 
    category_colnames = list(map(func, row)) 
    
    #rename columns
    categories.columns = category_colnames
    
    #iterate through each col. replace str with int
    # eg 'related-1' -> 1 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace(column+'-', '')
    
        # convert column from string to numeric
        categories[column] =  categories[column].astype(int)
        
    # new dataframe with dropped 'categories' column
    df_drop= df.drop('categories', axis=1)
    
    #concatenate df_drop with categories dataframe
    df_clean = pd.concat([df_drop, categories], axis=1)
    
    #drop duplicates
    df_clean.drop_duplicates(inplace=True)
    
    
    return df_clean


def save_data(df, database_filename):
    
    
    """
    DESCRIPTION: 
                Save processed data in a database table to be accessed 
                later on 
    
    INPUT: 
            df: (pandas.dataframe) processed dataset
            database_filename: (str) database to save table on 
        
    OUTPUT:
        None
    """
    # create engine for SQL database 
    engine = create_engine('sqlite:///'+ database_filename)
    
    # Save dataset under table name 'mescat'
    df.to_sql('mescat', engine, index=False)  


def main():
    ''' Main function to run ETL pipeline'''
    
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

