# import required modules
from operator import index
import numpy as np
from numpy.random import f
from sqlalchemy import create_engine
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv, find_dotenv
import re
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import requests

def get_averages(make, model, year):
    df = get_listings(make, model, year)

    if df.empty:
        return None

    numerical_cols = ['price', 'mileage', 'dom', 'dom_180']

    averages = df[numerical_cols].mean()
    
    return averages

def get_listings(make, model, year, host='localhost', port=5432):
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)

    make = make.strip()
    model = model.strip()
    year = int(year)

    table_name = os.environ.get('LISTINGS_TABLE')
    db_name = os.environ.get('DB_NAME')
    user = os.environ.get('DB_USER')
    password = os.environ.get('DB_PASSWORD')

    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')

    print(f"Table Name: {table_name}, Make: {make}, Model: {model}, Year: {year}")

    query = f"SELECT * FROM {table_name} WHERE make = '{make}' AND model = '{model}' AND year = {year};"
    df = pd.read_sql(query, engine)

    if df.empty:
        listings_to_sql(make, model, year)
        df = pd.read_sql(f"SELECT * FROM {table_name} WHERE make = '{make}' AND model = '{model}' AND year = {year};", engine)

    return df

def listings_to_sql(make, model, year, host='localhost', port=5432):
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)

    API_KEY = os.environ.get('MARKETCHECK_API_KEY')
    API_URL = f'https://mc-api.marketcheck.com/v2/search/car/active?api_key={API_KEY}&year={year}&make={make}&model={model}&include_relevant_links=true'

    response = requests.get(API_URL)

    if response.status_code == 200:
        table_name = str(os.environ.get('LISTINGS_TABLE'))
        db_name = os.environ.get('DB_NAME')
        user = os.environ.get('DB_USER')
        password = os.environ.get('DB_PASSWORD')

        engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')

        data = response.json()

        info = []

        listings = data.get('listings', [])
        for i, listing in enumerate(listings):
            price = listing.get('price')
            mileage = listing.get('miles')
            dom = listing.get('dom')
            dom_180 = listing.get('dom_180')

            info.append({
                'make': make,
                'model': model,
                'year': year,
                'price': price,
                'mileage': mileage,
                'dom': dom,
                'dom_180': dom_180
            })

        df = pd.DataFrame(info)

        df.to_sql(table_name, engine, if_exists='append', index=False)

# put cleaned data into the postgresql database
def to_sql(host='localhost', port=5432):
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)

    table_name = str(os.environ.get('CAR_DB_TABLE'))
    db_name = os.environ.get('DB_NAME')
    user = os.environ.get('DB_USER')
    password = os.environ.get('DB_PASSWORD')

    dfs = clean()

    if dfs is not None:
        engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
        for df in dfs:
            df = df.apply(lambda x: x.str.lower() if x.dtype == 'object' else x)
            df = df.drop_duplicates()
            df.to_sql(table_name, engine, if_exists='append', index=False)

# pull data from the postgresql database and prepare it for training
def to_pd(host='localhost', port=5432):
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)

    table_name = os.environ.get('CAR_DB_TABLE')
    db_name = os.environ.get('DB_NAME')
    user = os.environ.get('DB_USER')
    password = os.environ.get('DB_PASSWORD')

    categorical_cols = ['make', 'model']
    target_cols = ['price']

    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
    df = pd.read_sql(f'SELECT * FROM {table_name};', engine)

    df = clean_alphanumeric(df, categorical_cols)

    df_no_outliers = df[(df['price'] > 0) & (df['price'] < 20000)]
    target_df = df_no_outliers[target_cols]

    train_df, train_val_df, train_target, train_val_target = train_test_split(df_no_outliers, target_df, test_size=0.2, random_state=33)
    val_df, test_df, val_target, test_target = train_test_split(train_val_df, train_val_target, test_size=0.5, random_state=33)
    encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
    encoder.fit(df_no_outliers[categorical_cols])

    path = str(os.environ.get('ENCODER_PATH'))
    dump(encoder, path)

    train_encoded = encoder.transform(train_df[categorical_cols]).toarray()
    val_encoded = encoder.transform(val_df[categorical_cols]).toarray()
    test_encoded = encoder.transform(test_df[categorical_cols]).toarray()

    train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=train_df.index)
    val_encoded_df = pd.DataFrame(val_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=val_df.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=test_df.index)

    train_encoded_df = pd.concat([train_encoded_df, train_df[['year']].reset_index(drop=True)], axis=1)
    val_encoded_df = pd.concat([val_encoded_df, val_df[['year']].reset_index(drop=True)], axis=1)
    test_encoded_df = pd.concat([test_encoded_df, test_df[['year']].reset_index(drop=True)], axis=1)

    test_encoded_df = test_encoded_df.reindex(columns=train_encoded_df.columns, fill_value=0)

    return train_encoded_df, train_target, val_encoded_df, val_target, test_encoded_df, test_target

# clean the raw csv datasets from kaggle
def clean():
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)
    path1 = str(os.environ.get('DATA_PATH1'))
    path2 = str(os.environ.get('DATA_PATH2'))
    path3 = str(os.environ.get('DATA_PATH3'))
    path4 = str(os.environ.get('DATA_PATH4'))

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)
    df4 = pd.read_csv(path4)

    important_cols1 = ['Company', 'Model', 'Price ($)', 'Date']
    important_cols2 = ['year', 'make', 'model', 'sellingprice']
    important_cols3 = ['Make', 'Model', 'Year', 'MSRP']
    important_cols4 = ['price', 'year', 'manufacturer', 'model']

    df1.dropna(subset=important_cols1, inplace=True)
    df2.dropna(subset=important_cols2, inplace=True)
    df3.dropna(subset=important_cols3, inplace=True)
    df4.dropna(subset=important_cols4, inplace=True)

    df1 = df1[important_cols1]
    df2 = df2[important_cols2]
    df3 = df3[important_cols3]
    df4 = df4[important_cols4]

    df1['Date'] = pd.to_datetime(df1['Date'], format='%m/%d/%Y')
    df1['Year'] = df1['Date'].dt.year
    df1 = df1.drop('Date', axis=1)
    df4['year'] = df4['year'].round().astype(int)

    df1 = df1.rename(columns={'Company':'make', 'Model':'model', 'Price ($)':'price', 'Year':'year'})
    df2 = df2.rename(columns={'sellingprice':'price'})
    df3= df3.rename(columns={'Make':'make', 'Model':'model', 'Year':'year', 'MSRP':'price'})
    df4 = df4.rename(columns={'manufacturer':'make'})

    for df in [df1, df2, df3, df4]:
        df['make'] = df['make'].str.lower()
        df['model'] = df['model'].str.lower()

    df1_cleaned = df1[df1['make'].apply(lambda x: len(x) < 100)]
    df2_cleaned = df2[df2['make'].apply(lambda x: len(x) < 100)]
    df3_cleaned = df3[df3['make'].apply(lambda x: len(x) < 100)]
    df4_cleaned = df4[df4['make'].apply(lambda x: len(x) < 100)]

    df1_cleaned = df1[df1['model'].apply(lambda x: len(x) < 100)]
    df2_cleaned = df2[df2['model'].apply(lambda x: len(x) < 100)]
    df3_cleaned = df3[df3['model'].apply(lambda x: len(x) < 100)]
    df4_cleaned = df4[df4['model'].apply(lambda x: len(x) < 100)]

    return [df1_cleaned, df2_cleaned, df3_cleaned, df4_cleaned]

# cleaning helper function
def is_alphanumeric(s):
    return bool(re.match('^[a-zA-Z0-9]+$', s))

# cleaning helper function
def clean_alphanumeric(df, column_names):
    for column in column_names:
        # df[column] = df[column].astype(str)  
        df = df[df[column].apply(is_alphanumeric)]
    return df

# get the unique makes, model, and years for the drop downs
def uniques(host='localhost', port=5432):
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)

    table_name = os.environ.get('CAR_DB_TABLE')
    db_name = os.environ.get('DB_NAME')
    user = os.environ.get('DB_USER')
    password = os.environ.get('DB_PASSWORD')

    categorical_cols = ['make', 'model']

    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
    df = pd.read_sql(f'SELECT * FROM {table_name};', engine)

    df = clean_alphanumeric(df, categorical_cols)
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    df_no_outliers = df[(df['price'] > 0) & (df['price'] < 20000)]

    df_no_outliers.loc[df_no_outliers['model'] == 'alitma', 'model'] = 'altima'


    unique = []

    unique.append(sorted(df_no_outliers['make'].unique()))
    unique.append(sorted(df_no_outliers['model'].unique()))
    unique.append(sorted(df_no_outliers['year'].unique()))

    return unique

# verify that the user selects a valid make, model, and year
def verify(make, model, year, host='localhost', port=5432):
    ENV_FILE = find_dotenv()
    if ENV_FILE:
        load_dotenv(ENV_FILE)

    table_name = os.environ.get('CAR_DB_TABLE')
    db_name = os.environ.get('DB_NAME')
    user = os.environ.get('DB_USER')
    password = os.environ.get('DB_PASSWORD')

    engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')

    query = f"SELECT * FROM {table_name} WHERE make = %s AND model = %s AND year = %s"
    df = pd.read_sql(query, engine, params=(make, model, year))

    if df.empty:
        return False
    else:
        return True
