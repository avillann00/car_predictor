# import required modules
from django.shortcuts import render, redirect
from django.urls import reverse
from seaborn import categorical
from .forms import CarSearchForm
import ml_models.process
import ml_models.models
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import numpy as np
import pandas as pd

# render the landing page
def landing(request):
    return render(request, 'viz/landing.html', {'title': 'Landing'})

# render the search page
@login_required
def search(request): 
    messages.info(request, 'Please select a valid make, model, and year.')
    if request.method == 'GET':
        form = CarSearchForm(request.GET)
        if form.is_valid():
            make = form.cleaned_data.get('make')
            model = form.cleaned_data.get('model')
            year = form.cleaned_data.get('year')

            valid = ml_models.process.verify(make, model, year)

            if valid:
                prediction = perform_prediction(make, model, year)
                
                return redirect(reverse('car') + f'?prediction={prediction}')

            else:
                messages.error(request, 'Incorrect make, model, or year selected.')       

    else:
        form = CarSearchForm()

    return render(request, 'viz/search.html', {'form': form})

# render the car page after the prediction has been made
@login_required
def car(request):
    prediction = request.GET.get('prediction', None)
    return render(request, 'viz/car.html', {'prediction': prediction})

# function to transform the user input and make a prediction with the gradient boosting model
def perform_prediction(make, model, year):
    info = {'car_id': [1], 'make': [make], 'model': [model], 'year': [year], 'price': [0]}
    df = pd.DataFrame(info)
    categorical_cols = ['make', 'model']
    df['year'] = df['year'].astype('int')

    gb_model = ml_models.models.load_gb()
    encoder = ml_models.models.load_encoder()
    
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

