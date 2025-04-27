"""Views for the Featurama application."""

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpRequest, HttpResponse
from .models import Pipeline
from django.core.files.base import ContentFile
from .utils import (
    read_dataset_file, clean_up_temp_file, validate_dataset
)
from .services import MethodsService, PipelineResultsService, DatasetService
from .forms import (
    DatasetUploadForm, TargetVariableForm, 
    PipelineConfigForm, FeatureSelectionForm
)
from .algorithms import run_pipeline_analysis
import os
import pandas as pd


def pipelines(request: HttpRequest) -> HttpResponse:
    """List all pipelines or create a new one."""
    if request.method == 'POST':
        # Create a new pipeline with empty fields
        new_pipeline = Pipeline.objects.create()
        return redirect('featurama:upload_data', pipeline_id=new_pipeline.pk)

    pipelines = Pipeline.objects.all()  # Order by is now in Meta
    return render(
        request, 
        'featurama/pipelines.html', 
        {'pipelines': pipelines}
    )


def upload_data(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """Handle dataset upload and target variable selection."""
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    
    if request.method == 'POST':
        if 'dataset_file' in request.FILES:
            return _handle_file_upload(request, pipeline)
        elif 'target_variable' in request.POST:
            return _handle_target_selection(request, pipeline)
        elif 'selected_features' in request.POST:
            return _handle_feature_selection(request, pipeline)
    
    return render(
        request, 
        'featurama/upload_data.html', 
        {'pipeline': pipeline}
    )


def _handle_file_upload(
    request: HttpRequest, pipeline: Pipeline
) -> HttpResponse:
    """Process the uploaded dataset file."""
    form = DatasetUploadForm(request.POST, request.FILES)
    
    if not form.is_valid():
        return render(
            request, 
            'featurama/upload_data.html', 
            {
                'pipeline': pipeline,
                'error': form.errors['dataset_file'][0]
            }
        )
    
    try:
        dataset_file = form.cleaned_data['dataset_file']
        df, temp_file_path = read_dataset_file(dataset_file)
        
        # Store temp file info in session
        request.session['temp_file_path'] = temp_file_path
        request.session['temp_file_name'] = dataset_file.name

        # Get column names for target variable selection
        features = df.columns.tolist()
        dataset_name = os.path.splitext(dataset_file.name)[0]
        
        # Store features in session for later use
        request.session['dataset_features'] = features
        
        # Create form for target variable selection
        target_form = TargetVariableForm(features=features)
        
        return render(
            request, 
            'featurama/upload_data.html', 
            {
                'pipeline': pipeline,
                'features': features,
                'dataset_name': dataset_name,
                'target_form': target_form,
                'step': 'target_selection'
            }
        )
    except Exception as e:
        return render(
            request, 
            'featurama/upload_data.html',
            {
                'pipeline': pipeline,
                'error': f"Error reading file: {str(e)}"
            }
        )


def _handle_target_selection(
    request: HttpRequest, pipeline: Pipeline
) -> HttpResponse:
    """Process the target variable selection."""
    form = TargetVariableForm(request.POST)
    
    if not form.is_valid():
        return render(
            request,
            'featurama/upload_data.html',
            {
                'pipeline': pipeline,
                'error': "Please select a valid target variable."
            }
        )
    
    # Get the features from session
    features = request.session.get('dataset_features', [])
    if not features:
        return render(
            request,
            'featurama/upload_data.html',
            {
                'pipeline': pipeline,
                'error': "Session expired. Please upload the file again."
            }
        )
    
    # Get the selected target variable
    target_variable = form.cleaned_data['target_variable']
    
    # Store target variable in session
    request.session['target_variable'] = target_variable
    
    # Create form for feature selection
    feature_form = FeatureSelectionForm(
        features=features,
        target_variable=target_variable
    )
    
    return render(
        request,
        'featurama/upload_data.html',
        {
            'pipeline': pipeline,
            'features': features,
            'target_variable': target_variable,
            'feature_form': feature_form,
            'step': 'feature_selection'
        }
    )


def _handle_feature_selection(
    request: HttpRequest, pipeline: Pipeline
) -> HttpResponse:
    """Process feature selection and create dataset."""
    
    # Get data from session
    temp_file_path = request.session.get('temp_file_path')
    temp_file_name = request.session.get('temp_file_name')
    target_variable = request.session.get('target_variable')
    features = request.session.get('dataset_features', [])
    
    if not all([temp_file_path, temp_file_name, target_variable, features]):
        return render(
            request, 
            'featurama/upload_data.html', 
            {
                'pipeline': pipeline,
                'error': "Session expired. Please upload the file again."
            }
        )
    
    # Process the selected features
    selected_features = request.POST.getlist('selected_features')
    
    if not selected_features:
        return render(
            request,
            'featurama/upload_data.html',
            {
                'pipeline': pipeline,
                'features': features,
                'target_variable': target_variable,
                'feature_form': FeatureSelectionForm(
                    features=features,
                    target_variable=target_variable
                ),
                'step': 'feature_selection',
                'error': "Please select at least one feature."
            }
        )
    
    try:
        # Load the dataset for validation
        if temp_file_path.endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif temp_file_path.endswith('.xlsx'):
            # Use openpyxl for .xlsx files
            df = pd.read_excel(temp_file_path, engine='openpyxl')
        elif temp_file_path.endswith('.xls'):
            # Use xlrd for .xls files
            df = pd.read_excel(temp_file_path, engine='xlrd')
        else:
            # Get extension from original filename
            _, ext = os.path.splitext(temp_file_name.lower())
            if ext == '.csv':
                df = pd.read_csv(temp_file_path)
            elif ext == '.xlsx':
                df = pd.read_excel(temp_file_path, engine='openpyxl')
            elif ext == '.xls':
                df = pd.read_excel(temp_file_path, engine='xlrd')
            else:
                # Try to infer - this might still fail
                df = pd.read_csv(temp_file_path)
        
        # Validate dataset against requirements
        is_valid, error_message = validate_dataset(
            df, target_variable, selected_features
        )
        
        if not is_valid:
            return render(
                request,
                'featurama/upload_data.html',
                {
                    'pipeline': pipeline,
                    'features': features,
                    'target_variable': target_variable,
                    'feature_form': FeatureSelectionForm(
                        features=features,
                        target_variable=target_variable
                    ),
                    'step': 'feature_selection',
                    'error': error_message
                }
            )

        # Read the temporary file for dataset creation
        with open(temp_file_path, 'rb') as f:
            file_content = f.read()

        # Create dataset with the file, target variable, and selected features
        dataset = DatasetService.create_dataset(
            name=os.path.splitext(temp_file_name)[0],
            target_variable=target_variable,
            selected_features=selected_features
        )

        # Save the file to the dataset
        dataset.file.save(
            temp_file_name,
            ContentFile(file_content)
        )

        # Update pipeline with the new dataset
        pipeline.dataset = dataset
        pipeline.save()

        # Clean up
        clean_up_temp_file(request)
        
        # Clean up session
        if 'dataset_features' in request.session:
            del request.session['dataset_features']
        if 'target_variable' in request.session:
            del request.session['target_variable']
            
        return redirect(
            'featurama:configure_pipeline', 
            pipeline_id=pipeline.pk
        )
    except Exception as e:
        # Log the exact error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing file: {str(e)}\n{error_details}")
        
        return render(
            request, 
            'featurama/upload_data.html',
            {
                'pipeline': pipeline,
                'error': f"Error processing file: {str(e)}"
            }
        )


def configure_pipeline(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """Configure pipeline methods and parameters."""
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    
    # Check if the pipeline has been processed before
    # If it has all methods configured, we consider it as having results
    has_results = pipeline.is_configured()
    
    if request.method == 'POST':
        form = PipelineConfigForm(request.POST, instance=pipeline)
        if form.is_valid():
            form.save()
            
            # Run the complete pipeline analysis after saving the configuration
            run_pipeline_analysis(pipeline)
            
            return redirect(
                'featurama:results_summary', 
                pipeline_id=pipeline.pk
            )
        
        # If form is invalid, we'll render it with errors
        context = {
            'pipeline': pipeline,
            'form': form,
            'has_results': has_results,
            **MethodsService.get_available_methods()
        }
        return render(request, 'featurama/configure_pipeline.html', context)
    
    # GET request - show the form
    form = PipelineConfigForm(instance=pipeline)
    context = {
        'pipeline': pipeline,
        'form': form,
        'has_results': has_results,
        **MethodsService.get_available_methods()
    }
    
    return render(request, 'featurama/configure_pipeline.html', context)


def results_summary(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """Display the results summary for a pipeline."""
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    
    # Get all related data using our service
    related = PipelineResultsService.get_related_pipelines(pipeline)
    results = PipelineResultsService.get_pipeline_results_context(pipeline)
    
    context = {
        'pipeline': pipeline,
        'related_pipelines': related,
        **results
    }
    
    return render(request, 'featurama/results_summary.html', context)


def delete_pipeline(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """Delete a pipeline and redirect to the pipelines list view."""
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    
    # Delete all related objects that might have files
    # ShapExplanation objects will delete their image files
    shap_explanations = pipeline.shap_explanations.all()
    for shap in shap_explanations:
        shap.delete()
    
    # Delete feature selection results (no files to remove)
    pipeline.feature_selection_results.all().delete()
    
    # Delete performance metrics (no files to remove)
    pipeline.performance_metrics.all().delete()
    
    # Check if we should delete the dataset too
    if pipeline.dataset:
        other_pipelines = Pipeline.objects.filter(
            dataset=pipeline.dataset
        ).exclude(pk=pipeline.pk).count()
        
        if other_pipelines == 0:
            # Safe to delete the dataset as well (will delete its file)
            dataset = pipeline.dataset
            pipeline.delete()
            dataset.delete()
        else:
            # Just delete the pipeline
            pipeline.delete()
    else:
        pipeline.delete()
        
    return redirect('featurama:pipelines')
