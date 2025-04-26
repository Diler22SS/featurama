from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpRequest, HttpResponse
from .models import Pipeline, Dataset
import pandas as pd
import os
import tempfile
from django.core.files.base import ContentFile


# Create your views here.
def pipelines(request: HttpRequest) -> HttpResponse:
    if request.method == 'POST':
        # Create a new pipeline with empty fields
        new_pipeline = Pipeline.objects.create()
        return redirect('featurama:upload_data', pipeline_id=new_pipeline.id)

    pipelines = Pipeline.objects.all().order_by('-created_at')
    print(pipelines)
    return render(request, 'featurama/pipelines.html', {'pipelines': pipelines})


def upload_data(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    pipeline = get_object_or_404(Pipeline, id=pipeline_id)
    if request.method == 'POST':
        if 'dataset_file' in request.FILES:
            # First step: File upload
            dataset_file = request.FILES['dataset_file']
            # Read the file to get column names
            try:
                if dataset_file.name.endswith('.csv'):
                    df = pd.read_csv(dataset_file)
                else:  # Excel file
                    df = pd.read_excel(dataset_file)
                # Store the file temporarily
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                for chunk in dataset_file.chunks():
                    temp_file.write(chunk)
                temp_file.close()

                # Store temp file path in session
                request.session['temp_file_path'] = temp_file.name
                request.session['temp_file_name'] = dataset_file.name

                # Get column names for target variable selection
                features = df.columns.tolist()
                dataset_name = os.path.splitext(dataset_file.name)[0]
                return render(request, 'featurama/upload_data.html', {
                    'pipeline': pipeline,
                    'features': features,
                    'dataset_name': dataset_name
                })

            except Exception as e:
                return render(request, 'featurama/upload_data.html', {
                    'pipeline': pipeline,
                    'error': f"Error reading file: {str(e)}"
                })

        elif 'target_variable' in request.POST:
            # Second step: Target variable selection
            target_variable = request.POST['target_variable']

            # Get the temporary file path from session
            temp_file_path = request.session.get('temp_file_path')
            temp_file_name = request.session.get('temp_file_name')

            if not temp_file_path or not temp_file_name:
                return render(request, 'featurama/upload_data.html', {
                    'pipeline': pipeline,
                    'error': "File upload session expired. Please upload the file again."
                })

            try:
                # Read the temporary file
                with open(temp_file_path, 'rb') as f:
                    file_content = f.read()

                # Create dataset with the file and target variable
                dataset = Dataset.objects.create(
                    name=os.path.splitext(temp_file_name)[0],  # Use filename without extension
                    target_variable=target_variable
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
                os.unlink(temp_file_path)
                del request.session['temp_file_path']
                del request.session['temp_file_name']
                return redirect('featurama:configure_pipeline', pipeline_id=pipeline.id)
            except Exception as e:
                return render(request, 'featurama/upload_data.html', {
                    'pipeline': pipeline,
                    'error': f"Error processing file: {str(e)}"
                })
    return render(request, 'featurama/upload_data.html', {'pipeline': pipeline})


def configure_pipeline(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """Configure pipeline methods and parameters.
    
    Args:
        request: The HTTP request
        pipeline_id: ID of the pipeline to configure
        
    Returns:
        HttpResponse: Rendered configuration page or redirect
    """
    pipeline = get_object_or_404(Pipeline, id=pipeline_id)
    
    if request.method == 'POST':
        # Update pipeline with selected methods
        pipeline.filter_method = request.POST.get('filter_method')
        pipeline.wrapper_method = request.POST.get('wrapper_method')
        pipeline.model_method = request.POST.get('model_method')
        pipeline.save()
        return redirect('featurama:results_summary', pipeline_id=pipeline.id)
    
    # Define available methods with descriptions
    filter_methods = {
        'variance_threshold': 'Variance Threshold – removes low-variance features',
        'correlation': 'Correlation – removes highly correlated features',
        'mutual_info': 'Mutual Information – selects features based on mutual information score'
    }
    
    wrapper_methods = {
        'recursive_feature_elimination': 'Recursive Feature Elimination – recursively removes features',
        'sequential_feature_selection': 'Sequential Feature Selection – adds/removes features one at a time'
    }
    
    model_methods = {
        'random_forest': 'Random Forest – ensemble method using multiple decision trees',
        'xgboost': 'XGBoost – gradient boosting framework',
        'logistic_regression': 'Logistic Regression – linear model for classification'
    }
    
    return render(request, 'featurama/configure_pipeline.html', {
        'pipeline': pipeline,
        'filter_methods': filter_methods,
        'wrapper_methods': wrapper_methods,
        'model_methods': model_methods
    })


def delete_pipeline(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """Delete a pipeline and redirect to the pipelines list view.
    
    Args:
        request: The HTTP request
        pipeline_id: ID of the pipeline to delete
        
    Returns:
        HttpResponse: Redirect to pipelines list view
    """
    # TODO: Delete the dataset associated with the pipeline
    pipeline = get_object_or_404(Pipeline, id=pipeline_id)
    pipeline.delete()
    return redirect('featurama:pipelines')


def results_summary(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """Display the results summary for a pipeline.
    
    Args:
        request: The HTTP request
        pipeline_id: ID of the pipeline to show results for
        
    Returns:
        HttpResponse: Rendered results summary page
    """
    pipeline = get_object_or_404(Pipeline, id=pipeline_id)
    
    # Get all pipelines for the same dataset
    related_pipelines = Pipeline.objects.filter(
        dataset=pipeline.dataset
    ).exclude(id=pipeline.id).order_by('-created_at')
    
    return render(request, 'featurama/results_summary.html', {
        'pipeline': pipeline,
        'related_pipelines': related_pipelines,
        'metrics': {
            'roc_auc': None,  # Placeholder for actual metrics
            'accuracy': None,
            'f1': None
        },
        'selected_features': [],  # Placeholder for actual features
        'shap_plots': {
            'global': None,  # Placeholder for SHAP plots
            'local': None
        }
    })
