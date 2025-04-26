"""Views for the Featurama application."""

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpRequest, HttpResponse
from .models import Pipeline
from django.core.files.base import ContentFile
from .utils import read_dataset_file, clean_up_temp_file
from .services import MethodsService, PipelineResultsService, DatasetService
from .forms import DatasetUploadForm, TargetVariableForm, PipelineConfigForm
import os


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
        
        # Create form for target variable selection
        target_form = TargetVariableForm(features=features)
        
        return render(
            request, 
            'featurama/upload_data.html', 
            {
                'pipeline': pipeline,
                'features': features,
                'dataset_name': dataset_name,
                'target_form': target_form
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
    """Process the target variable selection and create dataset."""
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
    
    # Get the temporary file info from session
    temp_file_path = request.session.get('temp_file_path')
    temp_file_name = request.session.get('temp_file_name')

    if not temp_file_path or not temp_file_name:
        return render(
            request, 
            'featurama/upload_data.html', 
            {
                'pipeline': pipeline,
                'error': "File upload session expired. Please upload again."
            }
        )

    try:
        # Read the temporary file
        with open(temp_file_path, 'rb') as f:
            file_content = f.read()

        # Create dataset with the file and target variable
        target_variable = form.cleaned_data['target_variable']
        dataset = DatasetService.create_dataset(
            name=os.path.splitext(temp_file_name)[0],
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
        clean_up_temp_file(request)
        return redirect(
            'featurama:configure_pipeline', 
            pipeline_id=pipeline.pk
        )
    except Exception as e:
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
    
    if request.method == 'POST':
        form = PipelineConfigForm(request.POST, instance=pipeline)
        if form.is_valid():
            form.save()
            return redirect(
                'featurama:results_summary', 
                pipeline_id=pipeline.pk
            )
        
        # If form is invalid, we'll render it with errors
        context = {
            'pipeline': pipeline,
            'form': form,
            **MethodsService.get_available_methods()
        }
        return render(request, 'featurama/configure_pipeline.html', context)
    
    # GET request - show the form
    form = PipelineConfigForm(instance=pipeline)
    context = {
        'pipeline': pipeline,
        'form': form,
        **MethodsService.get_available_methods()
    }
    
    return render(request, 'featurama/configure_pipeline.html', context)


def results_summary(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """Display the results summary for a pipeline."""
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    
    # Get all related data using our service
    context = {
        'pipeline': pipeline,
        'related_pipelines': PipelineResultsService.get_related_pipelines(pipeline),
        **PipelineResultsService.get_pipeline_results_context(pipeline)
    }
    
    return render(request, 'featurama/results_summary.html', context)


def delete_pipeline(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """Delete a pipeline and redirect to the pipelines list view."""
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    
    # Delete associated dataset if no other pipelines use it
    if pipeline.dataset:
        other_pipelines = Pipeline.objects.filter(
            dataset=pipeline.dataset
        ).exclude(pk=pipeline.pk).count()
        
        if other_pipelines == 0:
            # Safe to delete the dataset as well
            dataset = pipeline.dataset
            pipeline.delete()
            dataset.delete()
        else:
            # Just delete the pipeline
            pipeline.delete()
    else:
        pipeline.delete()
        
    return redirect('featurama:pipelines')
