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
    PipelineConfigForm, FeatureSelectionForm,
    FilterMethodConfigForm, WrapperMethodConfigForm, ModelMethodConfigForm
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
        filter_form = FilterMethodConfigForm(request.POST)
        wrapper_form = WrapperMethodConfigForm(request.POST)
        model_form = ModelMethodConfigForm(request.POST)
        
        valid_forms = (form.is_valid() and filter_form.is_valid() and 
                      wrapper_form.is_valid() and model_form.is_valid())
        
        if valid_forms:
            # First save the basic pipeline configuration
            pipeline_instance = form.save(commit=False)
            
            # Get the filter method parameters with defaults
            filter_method = form.cleaned_data.get('filter_method')
            filter_params = {}
            
            # Add default values based on method
            if filter_method == 'Variance Threshold':
                # Default threshold is 0.1
                filter_params['threshold'] = filter_form.cleaned_data.get('threshold', 0.1)
            elif filter_method == 'ANOVA':
                # If k is not provided, use a sensible default (e.g., half of features)
                if filter_form.cleaned_data.get('k_anova'):
                    filter_params['k'] = filter_form.cleaned_data['k_anova']
            elif filter_method == 'Mutual Information':
                # If k is not provided, use a sensible default
                if filter_form.cleaned_data.get('k_mutual_info'):
                    filter_params['k'] = filter_form.cleaned_data['k_mutual_info']
            elif filter_method == 'MRMR':
                # Default n_features if not provided
                if filter_form.cleaned_data.get('n_features'):
                    filter_params['n_features'] = filter_form.cleaned_data['n_features']
                # Default method is 'MID'
                filter_params['method'] = filter_form.cleaned_data.get('method', 'MID')
            
            # Get the wrapper method parameters with defaults
            wrapper_method = form.cleaned_data.get('wrapper_method')
            wrapper_params = {}
            
            # Add default values based on method
            if wrapper_method == 'SFS with Logistic Regression':
                # Default scoring is 'accuracy'
                wrapper_params['scoring'] = wrapper_form.cleaned_data.get('scoring_logreg', 'accuracy')
            elif wrapper_method == 'SFS with Decision Tree':
                # Default scoring is 'accuracy'
                wrapper_params['scoring'] = wrapper_form.cleaned_data.get('scoring_tree', 'accuracy')
            elif wrapper_method == 'RFE with Logistic Regression':
                # Default n_features_to_select is 0.5 (50% of features)
                wrapper_params['n_features_to_select'] = wrapper_form.cleaned_data.get('n_features_logreg', 0.5)
            elif wrapper_method == 'RFE with Decision Tree':
                # Default n_features_to_select is 0.5 (50% of features)
                wrapper_params['n_features_to_select'] = wrapper_form.cleaned_data.get('n_features_tree', 0.5)
            
            # Get the model method parameters with defaults
            model_method = form.cleaned_data.get('model_method')
            model_params = {}
            
            # Add common default parameters
            model_params['test_size'] = model_form.cleaned_data.get('test_size', 0.25)
            
            # Add specific defaults based on method
            if model_method == 'Logistic Regression':
                model_params['C'] = model_form.cleaned_data.get('C', 1.0)
                model_params['penalty'] = model_form.cleaned_data.get('penalty', 'l2')
                model_params['solver'] = model_form.cleaned_data.get('solver', 'lbfgs')
                
            elif model_method in ['XGBoost Linear', 'XGBoost Tree']:
                model_params['n_estimators'] = model_form.cleaned_data.get('n_estimators', 100)
                model_params['learning_rate'] = model_form.cleaned_data.get('learning_rate', 0.3)
                
            elif model_method == 'Decision Tree':
                # Only add max_depth if explicitly provided
                if model_form.cleaned_data.get('max_depth'):
                    model_params['max_depth'] = model_form.cleaned_data['max_depth']
                
                model_params['min_samples_split'] = model_form.cleaned_data.get('min_samples_split', 0.01)
                model_params['min_samples_leaf'] = model_form.cleaned_data.get('min_samples_leaf', 0.01)
                model_params['criterion'] = model_form.cleaned_data.get('criterion', 'gini')
            
            # Set the parameters on the pipeline instance
            pipeline_instance.filter_params = filter_params
            pipeline_instance.wrapper_params = wrapper_params
            pipeline_instance.model_params = model_params
            pipeline_instance.save()
            
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
            'filter_form': filter_form,
            'wrapper_form': wrapper_form,
            'model_form': model_form,
            'has_results': has_results,
            **MethodsService.get_available_methods()
        }
        return render(request, 'featurama/configure_pipeline.html', context)
    
    # GET request - show the form
    form = PipelineConfigForm(instance=pipeline)
    
    # Create filter form and populate with existing params if available
    filter_form = FilterMethodConfigForm()
    if pipeline.filter_params:
        # Pre-populate form based on filter method
        if (pipeline.filter_method == 'Variance Threshold' and 
            'threshold' in pipeline.filter_params):
            filter_form.initial['threshold'] = pipeline.filter_params['threshold']
        
        elif pipeline.filter_method == 'ANOVA' and 'k' in pipeline.filter_params:
            filter_form.initial['k_anova'] = pipeline.filter_params['k']
        
        elif (pipeline.filter_method == 'Mutual Information' and 
              'k' in pipeline.filter_params):
            filter_form.initial['k_mutual_info'] = pipeline.filter_params['k']
        
        elif pipeline.filter_method == 'MRMR':
            if 'n_features' in pipeline.filter_params:
                filter_form.initial['n_features'] = pipeline.filter_params['n_features']
            if 'method' in pipeline.filter_params:
                filter_form.initial['method'] = pipeline.filter_params['method']
    
    # Create wrapper form and populate with existing params if available
    wrapper_form = WrapperMethodConfigForm()
    if pipeline.wrapper_params:
        # Pre-populate form based on wrapper method
        if (pipeline.wrapper_method == 'SFS with Logistic Regression' and 
            'scoring' in pipeline.wrapper_params):
            wrapper_form.initial['scoring_logreg'] = pipeline.wrapper_params['scoring']
        
        elif (pipeline.wrapper_method == 'SFS with Decision Tree' and 
              'scoring' in pipeline.wrapper_params):
            wrapper_form.initial['scoring_tree'] = pipeline.wrapper_params['scoring']
        
        elif (pipeline.wrapper_method == 'RFE with Logistic Regression' and 
              'n_features_to_select' in pipeline.wrapper_params):
            wrapper_form.initial['n_features_logreg'] = (
                pipeline.wrapper_params['n_features_to_select']
            )
        
        elif (pipeline.wrapper_method == 'RFE with Decision Tree' and 
              'n_features_to_select' in pipeline.wrapper_params):
            wrapper_form.initial['n_features_tree'] = (
                pipeline.wrapper_params['n_features_to_select']
            )
    
    # Create model form and populate with existing params if available
    model_form = ModelMethodConfigForm()
    if pipeline.model_params:
        # Common parameters
        if 'test_size' in pipeline.model_params:
            model_form.initial['test_size'] = pipeline.model_params['test_size']
            
        # Logistic Regression parameters
        if pipeline.model_method == 'Logistic Regression':
            if 'C' in pipeline.model_params:
                model_form.initial['C'] = pipeline.model_params['C']
            if 'penalty' in pipeline.model_params:
                model_form.initial['penalty'] = pipeline.model_params['penalty']
            if 'solver' in pipeline.model_params:
                model_form.initial['solver'] = pipeline.model_params['solver']
                
        # XGBoost parameters
        elif pipeline.model_method in ['XGBoost Linear', 'XGBoost Tree']:
            if 'n_estimators' in pipeline.model_params:
                model_form.initial['n_estimators'] = pipeline.model_params['n_estimators']
            if 'learning_rate' in pipeline.model_params:
                model_form.initial['learning_rate'] = pipeline.model_params['learning_rate']
                
        # Decision Tree parameters
        elif pipeline.model_method == 'Decision Tree':
            if 'max_depth' in pipeline.model_params:
                model_form.initial['max_depth'] = pipeline.model_params['max_depth']
            if 'min_samples_split' in pipeline.model_params:
                model_form.initial['min_samples_split'] = (
                    pipeline.model_params['min_samples_split']
                )
            if 'min_samples_leaf' in pipeline.model_params:
                model_form.initial['min_samples_leaf'] = (
                    pipeline.model_params['min_samples_leaf']
                )
            if 'criterion' in pipeline.model_params:
                model_form.initial['criterion'] = pipeline.model_params['criterion']
    
    context = {
        'pipeline': pipeline,
        'form': form,
        'filter_form': filter_form,
        'wrapper_form': wrapper_form,
        'model_form': model_form,
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
    styles['Heading3'].fontName = 'Arial'
    styles['Normal'].fontName = 'Arial'
    
    # Set font size
    styles['Title'].fontSize = 16
    styles['Heading1'].fontSize = 14
    styles['Heading2'].fontSize = 12
    styles['Heading3'].fontSize = 11
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
        f"Целевая переменная: {pipeline.dataset.target_variable}",
        f"Метод фильтрации: {pipeline.filter_method}",
        f"Метод обертки: {pipeline.wrapper_method}",
        f"Метод модели: {pipeline.model_method}"
    ]
    for item in info:
        story.append(Paragraph(item, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add method parameters
    story.append(Paragraph("Параметры методов", styles['Heading1']))
    
    # Filter method parameters
    if pipeline.filter_params:
        story.append(Paragraph(f"Параметры метода фильтрации ({pipeline.filter_method})", styles['Heading2']))
        
        # Create a table for better formatting
        param_description = {
            "threshold": "Порог дисперсии (признаки с меньшей дисперсией удаляются)",
            "k": "Количество лучших признаков для отбора",
            "n_features": "Количество признаков для отбора",
            "method": "Метод расчета mRMR (MID или MIQ)"
        }
        
        for param_name, param_value in pipeline.filter_params.items():
            desc = param_description.get(param_name, "Параметр метода фильтрации")
            story.append(Paragraph(f"• <b>{param_name}</b>: {param_value}", styles['Normal']))
            story.append(Paragraph(f"  <i>{desc}</i>", styles['Normal']))
            
        story.append(Spacer(1, 6))
    
    # Wrapper method parameters
    if pipeline.wrapper_params:
        story.append(Paragraph(f"Параметры метода обертки ({pipeline.wrapper_method})", styles['Heading2']))
        
        param_description = {
            "scoring": "Метрика оценки качества модели для отбора признаков",
            "n_features_to_select": "Доля или количество признаков для отбора"
        }
        
        for param_name, param_value in pipeline.wrapper_params.items():
            desc = param_description.get(param_name, "Параметр метода обертки")
            story.append(Paragraph(f"• <b>{param_name}</b>: {param_value}", styles['Normal']))
            story.append(Paragraph(f"  <i>{desc}</i>", styles['Normal']))
            
        story.append(Spacer(1, 6))
    
    # Model method parameters
    if pipeline.model_params:
        story.append(Paragraph(f"Параметры модели ({pipeline.model_method})", styles['Heading2']))
        
        param_description = {
            "test_size": "Доля данных для тестирования (от 0.01 до 0.99)",
            "C": "Параметр регуляризации (меньше = сильнее регуляризация)",
            "penalty": "Тип регуляризации (l1, l2, elasticnet, none)",
            "solver": "Алгоритм оптимизации",
            "n_estimators": "Количество деревьев в ансамбле",
            "learning_rate": "Скорость обучения (от 0.001 до 1.0)",
            "max_depth": "Максимальная глубина дерева",
            "min_samples_split": "Минимальная доля образцов для разделения узла",
            "min_samples_leaf": "Минимальная доля образцов в листе",
            "criterion": "Функция для измерения качества разделения"
        }
        
        for param_name, param_value in pipeline.model_params.items():
            desc = param_description.get(param_name, "Параметр модели")
            story.append(Paragraph(f"• <b>{param_name}</b>: {param_value}", styles['Normal']))
            story.append(Paragraph(f"  <i>{desc}</i>", styles['Normal']))
            
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
