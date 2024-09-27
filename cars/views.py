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
import requests
import os

# render the landing page
def landing(request):
    return render(request, 'cars/landing.html', {'title': 'Landing'})

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
                prediction = ml_models.models.perform_prediction_gb(make, model, year)
                averages = ml_models.process.get_averages(make, model, year)
                
                return redirect(reverse('car') + f"?prediction={prediction}&price={averages['price']}&mileage={averages['mileage']}&dom={averages['dom']}&dom_180={averages['dom_180']}")

            else:
                messages.error(request, 'Incorrect make, model, or year selected.')       

    else:
        form = CarSearchForm()

    return render(request, 'cars/search.html', {'form': form})

# render the car page after the prediction has been made
@login_required
def car(request):
    prediction = request.GET.get('prediction', None)
    avg_price = request.GET.get('price', None)
    avg_mileage = request.GET.get('mileage', None)
    avg_dom = request.GET.get('dom', None)
    avg_dom_180 = request.GET.get('dom_180', None)
    return render(request, 'cars/car.html', {
        'prediction': prediction,
        'avg_price': avg_price,
        'avg_mileage': avg_mileage,
        'avg_dom': avg_dom,
        'avg_dom_180': avg_dom_180
    })
