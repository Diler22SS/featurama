"""Forms for the Featurama application.

This module defines Django forms for validating user input.
"""

from django import forms
from .models import Pipeline
from .services import MethodsService


class DatasetUploadForm(forms.Form):
    """Form for validating dataset file uploads."""
    
    ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'xls']
    
    dataset_file = forms.FileField(
        required=True,
        widget=forms.ClearableFileInput(
            attrs={'accept': '.csv,.xlsx,.xls'}
        )
    )
    
    def clean_dataset_file(self):
        """Validate the uploaded file."""
        file = self.cleaned_data['dataset_file']
        
        # Check file extension
        extension = file.name.split('.')[-1].lower()
        if extension not in self.ALLOWED_EXTENSIONS:
            raise forms.ValidationError(
                f"Unsupported file format. Please upload one of: "
                f"{', '.join(self.ALLOWED_EXTENSIONS)}."
            )
            
        # Check file size (10MB max)
        if file.size > 10 * 1024 * 1024:
            raise forms.ValidationError(
                "File too large. Maximum size is 10MB."
            )
            
        return file


class TargetVariableForm(forms.Form):
    """Form for selecting a target variable."""
    
    target_variable = forms.CharField(
        required=True,
        widget=forms.Select()
    )
    
    def __init__(self, *args, features=None, **kwargs):
        """Initialize the form with dynamic choices."""
        super().__init__(*args, **kwargs)
        
        if features:
            self.fields['target_variable'].widget.choices = [
                ('', 'Select a target variable')
            ] + [(f, f) for f in features]


class FeatureSelectionForm(forms.Form):
    """Form for selecting features from a dataset."""
    
    selected_features = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple(
            attrs={'class': 'feature-checkbox'}
        )
    )
    
    def __init__(self, *args, features=None, target_variable=None, **kwargs):
        """Initialize the form with available features.
        
        Args:
            features: List of all available features in the dataset
            target_variable: Target variable to exclude from selection options
        """
        super().__init__(*args, **kwargs)
        
        if features and target_variable:
            # Filter out the target variable from selectable features
            available_features = [f for f in features if f != target_variable]
            choices = [(f, f) for f in available_features]
            self.fields['selected_features'].choices = choices
            
            # Store original features for reference
            self.all_features = features
            self.target_variable = target_variable


class PipelineConfigForm(forms.ModelForm):
    """Form for configuring pipeline methods."""
    
    class Meta:
        model = Pipeline
        fields = ['filter_method', 'wrapper_method', 'model_method']
        
    def __init__(self, *args, **kwargs):
        """Initialize the form with dynamic choices."""
        super().__init__(*args, **kwargs)
        
        # Get available methods
        methods = MethodsService.get_available_methods()
        
        # Set choices for each method field
        self.fields['filter_method'].widget = forms.Select(choices=[
            ('', 'Select a filter method')
        ] + [(k, k.replace('_', ' ').title()) 
             for k in methods['filter_methods']])
        
        self.fields['wrapper_method'].widget = forms.Select(choices=[
            ('', 'Select a wrapper method')
        ] + [(k, k.replace('_', ' ').title()) 
             for k in methods['wrapper_methods']])
        
        self.fields['model_method'].widget = forms.Select(choices=[
            ('', 'Select a model method')
        ] + [(k, k.replace('_', ' ').title()) 
             for k in methods['model_methods']])
        
        # Mark all fields as required
        for field in self.fields.values():
            field.required = True 