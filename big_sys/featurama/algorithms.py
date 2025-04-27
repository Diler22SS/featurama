"""Feature selection algorithms for the Featurama application.

This module contains algorithms used for feature selection in pipelines.
"""

import random
from .models import Pipeline, FeatureSelectionResult


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


def run_feature_selection(pipeline: Pipeline) -> FeatureSelectionResult:
    """Run the appropriate feature selection algorithm based on pipeline config.
    
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