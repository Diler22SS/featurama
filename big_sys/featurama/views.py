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
                return redirect('featurama:pipelines')
            except Exception as e:
                return render(request, 'featurama/upload_data.html', {
                    'pipeline': pipeline,
                    'error': f"Error processing file: {str(e)}"
                })
    return render(request, 'featurama/upload_data.html', {'pipeline': pipeline})


def delete_pipeline(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    pipeline = get_object_or_404(Pipeline, id=pipeline_id)
    pipeline.delete()
    return redirect('featurama:pipelines')
