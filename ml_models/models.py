# import required modules
from sqlalchemy import exists
from xgboost import XGBRegressor, train
from sklearn.linear_model import LinearRegression
from . import process
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd

# load the one hot encoder
def load_encoder():
    ENV_FILE = find_dotenv()
    if ENV_FILE:
            load_dotenv(ENV_FILE)

    path = str(os.environ.get('ENCODER_PATH'))
    return load(path)

# load the gradient boosting model
def load_gb():
    ENV_FILE = find_dotenv()
    if ENV_FILE:
            load_dotenv(ENV_FILE)

    path = str(os.environ.get('GB_PATH'))
    return load(path)

# load the linear model
def load_linear():
    ENV_FILE = find_dotenv()
    if ENV_FILE:
            load_dotenv(ENV_FILE)

    path = str(os.environ.get('LIN_PATH'))
    return load(path)

# save the gradient boosting model
def save_gb(): 
    ENV_FILE = find_dotenv()
    if ENV_FILE:
            load_dotenv(ENV_FILE)

    model = gb_model()
    path = str(os.environ.get('GB_PATH'))

    dump(model, path)

# save the linear model
def save_linear():
    ENV_FILE = find_dotenv()
    if ENV_FILE:
            load_dotenv(ENV_FILE)

    model = linear_model()
    path = str(os.environ.get('LIN_PATH'))

    dump(model, path)

# instantiate and train the gradient boosting model
def gb_model():
    train_df, train_target, val_df, val_target, test_df, test_target = process.to_pd()

    model = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=1000, learning_rate=0.3, max_depth=10)
    model.fit(train_df, train_target)

    return model

# instantiate and train the gradient boosting model
def linear_model():
    train_df, train_target, val_df, val_target, test_df, test_target = process.to_pd()
    train_df = train_df.iloc[:len(train_df)//2]
    train_target = train_target[:len(train_target)//2]

    model = LinearRegression()
    model.fit(train_df, train_target)

    return model

# function to transform the user input and make a prediction with the gradient boosting model
def perform_prediction_gb(make, model, year):
    info = {'car_id': [1], 'make': [make], 'model': [model], 'year': [year], 'price': [0]}
    df = pd.DataFrame(info)
    categorical_cols = ['make', 'model']
    df['year'] = df['year'].astype('int')

    gb_model = load_gb()
    encoder = load_encoder()
    
    encoded = encoder.transform(df[categorical_cols]).toarray()
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    encoded_df = pd.concat([encoded_df, df[['year']].reset_index(drop=True)], axis=1)

    required_columns = gb_model.get_booster().feature_names
    missing_cols = [col for col in required_columns if col not in encoded_df.columns]

    for col in missing_cols:
        encoded_df[col] = 0

    encoded_df = encoded_df[required_columns]

    prediction = gb_model.predict(encoded_df)
    return round(prediction[0], 2)

