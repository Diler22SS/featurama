"""Feature selection algorithms for the Featurama application.

This module contains algorithms used for feature selection in pipelines.
"""

# Standard library imports
import io
import random

# Third-party imports
# Configure matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from django.core.files.base import ContentFile

# Local application imports
from .models import (
    Pipeline, FeatureSelectionResult, 
    PerformanceMetric, ShapExplanation
)


def random_feature_selection(pipeline: Pipeline) -> FeatureSelectionResult:
    """Randomly select features from the user selected features.
    
    This function creates a FeatureSelectionResult object with randomly
    selected features from the user_selected_features in the dataset.
    
    Args:
        pipeline: The pipeline for which to perform feature selection
        
    Returns:
        A FeatureSelectionResult object with the selected features
    """
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return FeatureSelectionResult.objects.create(
            pipeline=pipeline,
            selected_features=[]
        )
    
    # Get user selected features
    all_features = pipeline.dataset.user_selected_features
    
    # Select a random subset (between 30% and 70% of features)
    num_to_select = max(1, random.randint(
        int(len(all_features) * 0.3), 
        int(len(all_features) * 0.7)
    ))
    
    # Randomly select features
    selected = random.sample(all_features, num_to_select)
    
    # Create and return the result
    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        selected_features=selected
    )


def evaluate_model(pipeline: Pipeline) -> PerformanceMetric:
    """Generate random performance metrics for the model.
    
    This is a placeholder function that generates random performance metrics
    for demonstration purposes. In a real implementation, this would run
    the actual model evaluation.
    
    Args:
        pipeline: The pipeline for which to generate metrics
        
    Returns:
        A PerformanceMetric object with random values
    """
    # Generate random metrics (values between 0.7 and 1.0 for demo purposes)
    roc_auc = round(random.uniform(0.7, 1.0), 3)
    accuracy = round(random.uniform(0.7, 1.0), 3)
    f1_score = round(random.uniform(0.7, 1.0), 3)
    
    # Create and return the metrics
    return PerformanceMetric.objects.create(
        pipeline=pipeline,
        roc_auc=roc_auc,
        accuracy=accuracy,
        f1_score=f1_score
    )


def generate_shap_plots(pipeline: Pipeline) -> ShapExplanation:
    """Generate random SHAP explanation plots.
    
    This is a placeholder function that generates random SHAP plots
    for demonstration purposes. In a real implementation, this would generate
    actual SHAP plots based on the model and data.
    
    Args:
        pipeline: The pipeline for which to generate SHAP plots
        
    Returns:
        A ShapExplanation object with the generated plot images
    """
    # Create a new ShapExplanation object
    shap_explanation = ShapExplanation.objects.create(
        pipeline=pipeline
    )
    
    # Generate and save global explanation plot
    global_plot = _generate_global_shap_plot(pipeline)
    shap_explanation.global_explanation_image.save(
        f'global_shap_{pipeline.id}.png',
        ContentFile(global_plot),
        save=True
    )
    
    # Generate and save local explanation plot
    local_plot = _generate_local_shap_plot(pipeline)
    shap_explanation.local_explanation_image.save(
        f'local_shap_{pipeline.id}.png',
        ContentFile(local_plot),
        save=True
    )
    
    return shap_explanation


def _generate_global_shap_plot(pipeline: Pipeline) -> bytes:
    """Generate a random global SHAP plot.
    
    Args:
        pipeline: The pipeline to generate the plot for
        
    Returns:
        The plot as bytes
    """
    plt.figure(figsize=(10, 6))
    
    # Get feature names from the pipeline
    features = []
    if pipeline.dataset and pipeline.dataset.user_selected_features:
        features = pipeline.dataset.user_selected_features
    else:
        # Use generic feature names if no real features available
        features = [f'Feature {i}' for i in range(10)]
    
    # Generate random importance values
    importances = [random.uniform(0, 1) for _ in range(len(features))]
    
    # Sort features by importance
    sorted_indices = np.argsort(importances)
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]
    
    # Create horizontal bar plot
    plt.barh(sorted_features, sorted_importances, color='skyblue')
    plt.xlabel('SHAP Value (Impact on Model Output)')
    plt.title('Global Feature Importance')
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return buf.getvalue()


def _generate_local_shap_plot(pipeline: Pipeline) -> bytes:
    """Generate a random local SHAP plot.
    
    Args:
        pipeline: The pipeline to generate the plot for
        
    Returns:
        The plot as bytes
    """
    plt.figure(figsize=(10, 6))
    
    # Get feature names from the pipeline
    features = []
    if pipeline.dataset and pipeline.dataset.user_selected_features:
        # Take fewer features for local explanation (max 5)
        features = pipeline.dataset.user_selected_features[:5]
    else:
        # Use generic feature names if no real features available
        features = [f'Feature {i}' for i in range(5)]
    
    # Generate random SHAP values (both positive and negative)
    shap_values = [random.uniform(-1, 1) for _ in range(len(features))]
    
    # Create colors based on value sign
    colors = ['red' if val < 0 else 'blue' for val in shap_values]
    
    # Create horizontal bar plot
    plt.barh(features, shap_values, color=colors)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel('SHAP Value')
    plt.title('Local Feature Impact (Sample Prediction)')
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return buf.getvalue()


def run_feature_selection(pipeline: Pipeline) -> FeatureSelectionResult:
    """Run the appropriate feature selection algorithm.
    
    This function determines which feature selection algorithm to run based on
    the pipeline's configuration and returns the result.
    
    Args:
        pipeline: The pipeline for which to perform feature selection
        
    Returns:
        A FeatureSelectionResult object with the selected features
    """
    # For now, just use random selection regardless of configuration
    # In the future, this would dispatch to different algorithms based on
    # the pipeline's filter_method, wrapper_method, and model_method
    return random_feature_selection(pipeline)


def run_pipeline_analysis(pipeline: Pipeline) -> None:
    """Run all pipeline analysis steps.
    
    This function runs feature selection, model evaluation, and SHAP analysis
    for a pipeline. It's a comprehensive function that can be called after
    pipeline configuration is complete.
    
    Args:
        pipeline: The pipeline to analyze
    """
    # Step 1: Run feature selection
    run_feature_selection(pipeline)
    
    # Step 2: Evaluate the model
    evaluate_model(pipeline)
    
    # Step 3: Generate SHAP explanations
    generate_shap_plots(pipeline) 