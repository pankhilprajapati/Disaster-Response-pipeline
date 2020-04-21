import sys
import pandas as pd
from sqlalchemy import *

def load_data(messages_filepath, categories_filepath):
    '''
    load the data set from the csv file and convert it to pandas
    dataframe and combine the two data frame

    Argument :
         messages_filepath - path of the csv file disaster_messages.csv

         categories_filepath - path of the csv file disaster_categories.csv

    return : 
         df - uncleaned data frame

    '''  

    # load messages and categories datasets
    messages = pd.read_csv('disaster_messages.csv')
    categories = pd.read_csv('disaster_categories.csv')
    
    # merge datasets
    df = pd.merge(messages,categories, on="id")
    
    return df

def clean_data(df):
    '''
    This function takes the Dataframe and clean the it like
    removing the duplicate variable and slicing the unnecessary
    data, spliting it. Also make clean target variables

    Argument:
         df - takes the dataframe and clean it
    return:
         df - cleaned dataframe
    '''

    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(';',expand = True))

    # select the first row of the categories dataframe and use row to 
    # to get the last element of the string in the column
    row = list(categories.iloc[0])
    category_colnames = list(map(lambda x: x.split('-')[0],row))
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str.get(-1)
        categories[column] = pd.to_numeric(categories[column],errors='coerce')

    df = df.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    '''
    Save the clean data in to sql database using pd.to_sql()

    Argument:
         df - clean data frame
         database_filename - name of the databasefile 
    
    return :
         None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename.split('.')[0], engine, index=False,if_exists = 'replace')

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