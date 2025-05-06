"""Views for the Featurama application."""

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpRequest, HttpResponse
from django.db import models
from django.conf import settings

from .models import Pipeline, FeatureSelectionResult, ShapExplanation
from .utils import (
    read_dataset_file, validate_dataset
)
from .services import MethodsService, PipelineResultsService, DatasetService
from .forms import (
    DatasetUploadForm, TargetVariableForm, 
    PipelineConfigForm, FeatureSelectionForm
)
from .algorithms import run_pipeline
import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


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
        # Check if user clicked a back button
        if 'go_back' in request.POST:
            if request.POST['go_back'] == 'upload':
                # Go back to file upload step
                return render(
                    request, 
                    'featurama/upload_data.html', 
                    {'pipeline': pipeline}
                )
            elif request.POST['go_back'] == 'target':
                # Go back to target selection step
                if 'temp_file_path' not in request.session:
                    # If session expired, go back to upload
                    return render(
                        request, 
                        'featurama/upload_data.html', 
                        {'pipeline': pipeline}
                    )
                
                # Get data from session
                features = request.session.get('dataset_features', [])
                binary_vars = request.session.get('binary_variables', [])
                temp_file_name = request.session.get('temp_file_name', '')
                dataset_name = os.path.splitext(temp_file_name)[0]
                
                # Create form for target variable selection
                target_form = TargetVariableForm(features=features)
                
                return render(
                    request, 
                    'featurama/upload_data.html', 
                    {
                        'pipeline': pipeline,
                        'features': features,
                        'binary_variables': binary_vars,
                        'dataset_name': dataset_name,
                        'target_form': target_form,
                        'step': 'target_selection'
                    }
                )
        
        # Normal flow
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
        
        # Store data in session
        request.session['dataset_data'] = df.to_json(orient='records')
        request.session['dataset_name'] = dataset_file.name

        # Get column names for target variable selection
        features = df.columns.tolist()
        dataset_name = os.path.splitext(dataset_file.name)[0]
        
        # Identify binary variables (columns with only 2 unique values)
        binary_vars = []
        for col in features:
            try:
                if df[col].nunique() == 2:
                    binary_vars.append(col)
            except Exception:
                # Skip columns that can't be analyzed (e.g., complex objects)
                pass
        
        # Store features and binary vars in session for later use
        request.session['dataset_features'] = features
        request.session['binary_variables'] = binary_vars
        
        # Create form for target variable selection
        target_form = TargetVariableForm(features=features)
        
        return render(
            request, 
            'featurama/upload_data.html', 
            {
                'pipeline': pipeline,
                'features': features,
                'binary_variables': binary_vars,
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
                'error': f"Ошибка чтения файла: {str(e)}"
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
                'error': "Пожалуйста, выберите корректную целевую переменную."
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
                'error': "Сессия истекла. Пожалуйста, загрузите файл снова."
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
    dataset_data = request.session.get('dataset_data')
    dataset_name = request.session.get('dataset_name')
    target_variable = request.session.get('target_variable')
    features = request.session.get('dataset_features', [])
    
    if not all([dataset_data, dataset_name, target_variable, features]):
        return render(
            request, 
            'featurama/upload_data.html', 
            {
                'pipeline': pipeline,
                'error': "Сессия истекла. Пожалуйста, загрузите файл снова."
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
                'error': "Пожалуйста, выберите хотя бы один признак."
            }
        )
    
    try:
        # Load the dataset from JSON
        df = pd.read_json(dataset_data)
        
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

        # Create dataset with the data, target variable, and selected features
        dataset = DatasetService.create_dataset(
            name=os.path.splitext(dataset_name)[0],
            target_variable=target_variable,
            selected_features=selected_features
        )

        # Save the DataFrame as JSON
        dataset.save_dataframe(df)

        # Update pipeline with the new dataset
        pipeline.dataset = dataset
        pipeline.save()
        
        # Clean up session
        if 'dataset_data' in request.session:
            del request.session['dataset_data']
        if 'dataset_name' in request.session:
            del request.session['dataset_name']
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
        print(f"Ошибка обработки файла: {str(e)}\n{error_details}")
        
        return render(
            request, 
            'featurama/upload_data.html',
            {
                'pipeline': pipeline,
                'error': f"Ошибка обработки файла: {str(e)}"
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
            
            # Run filter and wrapper methods only
            run_pipeline(pipeline, run_model=False)
            
            return redirect(
                'featurama:manual_feature_selection', 
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


def manual_feature_selection(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """Allow user to manually select features after filter and wrapper methods."""
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    
    # Get the latest feature selection result
    result = FeatureSelectionResult.objects.filter(pipeline=pipeline).first()
    
    if not result:
        # If we have no results, redirect to the configuration page
        return redirect('featurama:configure_pipeline', pipeline_id=pipeline.pk)
    
    # Get original user selected features
    user_features = pipeline.dataset.user_selected_features or []
    
    # Get filtered and wrapped features from the most recent result
    filtered_features = result.filtered_features or []
    wrapped_features = result.wrapped_features or []
    
    # For the initial display, use wrapped features if available, otherwise use filtered
    selected_features = wrapped_features if wrapped_features else filtered_features
    
    if request.method == 'POST':
        # Process manual feature selection
        manually_selected = request.POST.getlist('selected_features')
        print(f"Manually selected features: {len(manually_selected)} \n {manually_selected}")
        
        if not manually_selected:
            # Require at least one feature
            return render(
                request, 
                'featurama/manual_feature_selection.html',
                {
                    'pipeline': pipeline,
                    'user_features': user_features,
                    'filtered_features': filtered_features,
                    'wrapped_features': wrapped_features,
                    'selected_features': selected_features,
                    'error': 'Пожалуйста, выберите хотя бы один признак'
                }
            )
        
        # Save manually selected features
        result.manual_features = manually_selected
        result.save()
        
        # Now run the model step
        run_pipeline(pipeline)
        
        return redirect('featurama:results_summary', pipeline_id=pipeline.pk)
    
    # GET request - show the form
    return render(
        request, 
        'featurama/manual_feature_selection.html',
        {
            'pipeline': pipeline,
            'user_features': user_features,
            'filtered_features': filtered_features,
            'wrapped_features': wrapped_features,
            'selected_features': selected_features
        }
    )


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


def export_report(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """Export pipeline results as PDF report."""
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    
    # Get all related data
    results = PipelineResultsService.get_pipeline_results_context(pipeline)
    
    # Create PDF response
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="pipeline_{pipeline_id}_report.pdf"'
    
    # Create PDF document
    doc = SimpleDocTemplate(response, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Configure fonts for Cyrillic support
    pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
    
    # Configure styles
    styles['Title'].fontName = 'Arial'
    styles['Heading1'].fontName = 'Arial'
    styles['Heading2'].fontName = 'Arial'
    styles['Normal'].fontName = 'Arial'
    
    # Set font size
    styles['Title'].fontSize = 16
    styles['Heading1'].fontSize = 14
    styles['Heading2'].fontSize = 12
    styles['Normal'].fontSize = 10
    
    story = []
    
    # Add title
    title = Paragraph(f"Отчет по пайплайну #{pipeline_id}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Add pipeline info
    story.append(Paragraph("Информация о пайплайне", styles['Heading1']))
    info = [
        f"Набор данных: {pipeline.dataset.name}",
        f"Метод фильтрации: {pipeline.filter_method}",
        f"Метод обертки: {pipeline.wrapper_method}",
        f"Метод модели: {pipeline.model_method}"
    ]
    for item in info:
        story.append(Paragraph(item, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add metrics
    story.append(Paragraph("Метрики модели", styles['Heading1']))
    metrics = [
        f"ROC-AUC: {results['metrics']['roc_auc']}",
        f"Точность: {results['metrics']['accuracy']}",
        f"F1 оценка: {results['metrics']['f1']}"
    ]
    for item in metrics:
        story.append(Paragraph(item, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add features
    story.append(Paragraph("Выбранные признаки", styles['Heading1']))
    story.append(Paragraph("Начальные признаки:", styles['Heading2']))
    for feature in results['user_selected_features']:
        story.append(Paragraph(f"- {feature}", styles['Normal']))
    story.append(Spacer(1, 6))
    
    story.append(Paragraph("Выбранные признаки:", styles['Heading2']))
    for feature in results['selected_features']:
        story.append(Paragraph(f"- {feature}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add SHAP Analysis
    story.append(Paragraph("SHAP анализ", styles['Heading1']))
    
    # Get the latest SHAP explanation
    try:
        shap_explanation = ShapExplanation.objects.filter(
            pipeline=pipeline
        ).order_by('-created_at').first()
        
        if shap_explanation:
            # Add Global SHAP Plot
            story.append(Paragraph("Глобальная важность признаков", styles['Heading2']))
            if shap_explanation.global_explanation_image:
                try:
                    img = Image(
                        shap_explanation.global_explanation_image.path,
                        width=400,
                        height=300
                    )
                    story.append(img)
                except Exception as e:
                    story.append(Paragraph(
                        f"Ошибка загрузки глобального SHAP графика: {str(e)}",
                        styles['Normal']
                    ))
            else:
                story.append(Paragraph(
                    "Нет доступного глобального SHAP графика",
                    styles['Normal']
                ))
            story.append(Spacer(1, 12))
            
            # Add Distribution SHAP Plot 
            story.append(Paragraph("Распределение важности признаков", styles['Heading2']))
            if shap_explanation.distribution_explanation_image:
                try:
                    img = Image(
                        shap_explanation.distribution_explanation_image.path,
                        width=400,
                        height=300
                    )
                    story.append(img)
                except Exception as e:
                    story.append(Paragraph(
                        f"Ошибка загрузки графика распределения SHAP: {str(e)}",
                        styles['Normal']
                    ))
            else:
                story.append(Paragraph(
                    "Нет доступного графика распределения SHAP",
                    styles['Normal']
                ))
        else:
            story.append(Paragraph(
                "Нет доступного SHAP анализа для этого пайплайна",
                styles['Normal']
            ))
    except Exception as e:
        story.append(Paragraph(
            f"Ошибка доступа к SHAP анализу: {str(e)}",
            styles['Normal']
        ))
    
    # Build PDF
    doc.build(story)
    
    return response
