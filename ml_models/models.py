# import required modules
from sqlalchemy import exists
from xgboost import XGBRegressor, train
from sklearn.linear_model import LinearRegression
from . import process
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load
import os
from dotenv import load_dotenv, find_dotenv

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
