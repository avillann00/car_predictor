# import required modules
from django import forms
import ml_models.process

# get the unique options so the form can populate them
unique_makes, unique_models, unique_years = ml_models.process.uniques()

# custom form so the user can select a car
class CarSearchForm(forms.Form):
    make = forms.ChoiceField(
        choices=[(make, make) for make in unique_makes],
        required=True,
        label='Select Make',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    model = forms.ChoiceField(
        choices=[(model, model) for model in unique_models],
        required=True,
        label='Select Model',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    year = forms.ChoiceField(
        choices=[(str(year), str(year)) for year in unique_years],
        required=True,
        label='Select Year',
        widget=forms.Select(attrs={'class': 'form-control'})
    )

