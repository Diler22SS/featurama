"""Service layer for Featurama business logic.

This module contains classes that handle the business logic for the application,
separating it from the view functions and keeping controllers thin.
"""

from typing import Dict, List, Optional, Any
from .models import Pipeline, Dataset


class MethodsService:
    """Service for managing pipeline methods and their descriptions."""
    
    @staticmethod
    def get_available_methods() -> Dict[str, Dict[str, str]]:
        """Get all available pipeline methods with descriptions.
        
        Returns:
            Dictionary containing all methods with descriptions
        """
        filter_methods = {
            'variance_threshold': 'Variance Threshold – removes low-variance features',
            'correlation': 'Correlation – removes highly correlated features',
            'mutual_info': 'Mutual Information – selects features based on mutual info'
        }
        
        wrapper_methods = {
            'recursive_feature_elimination': 'Recursive Feature Elimination',
            'sequential_feature_selection': 'Sequential Feature Selection'
        }
        
        model_methods = {
            'random_forest': 'Random Forest – ensemble using multiple decision trees',
            'xgboost': 'XGBoost – gradient boosting framework',
            'logistic_regression': 'Logistic Regression – linear model for classification'
        }
        
        return {
            'filter_methods': filter_methods,
            'wrapper_methods': wrapper_methods,
            'model_methods': model_methods
        }


class PipelineResultsService:
    """Service for generating pipeline results and metrics."""
    
    @staticmethod
    def get_pipeline_metrics(pipeline: Pipeline) -> Dict[str, Optional[float]]:
        """Get metrics for a pipeline.
        
        Args:
            pipeline: The pipeline object
            
        Returns:
            Dictionary with metrics
        """
        # Placeholder for actual metrics calculation
        return {
            'roc_auc': None,
            'accuracy': None,
            'f1': None
        }
    
    @staticmethod
    def get_selected_features(pipeline: Pipeline) -> List[str]:
        """Get selected features for a pipeline.
        
        Args:
            pipeline: The pipeline object
            
        Returns:
            List of selected feature names
        """
        if pipeline.dataset and pipeline.dataset.user_selected_features:
            return pipeline.dataset.user_selected_features
        return []
    
    @staticmethod
    def get_shap_plots(pipeline: Pipeline) -> Dict[str, Optional[str]]:
        """Get SHAP plots for a pipeline.
        
        Args:
            pipeline: The pipeline object
            
        Returns:
            Dictionary with plot URLs
        """
        # Placeholder for actual SHAP plot generation
        return {
            'global': None,
            'local': None
        }
    
    @classmethod
    def get_pipeline_results_context(cls, pipeline: Pipeline) -> Dict[str, Any]:
        """Get all results data for a pipeline in a format ready for templates.
        
        This method aggregates all results data for a pipeline for easy use
        in views.
        
        Args:
            pipeline: The pipeline object
            
        Returns:
            Context dictionary with all results data
        """
        return {
            'metrics': cls.get_pipeline_metrics(pipeline),
            'selected_features': cls.get_selected_features(pipeline),
            'shap_plots': cls.get_shap_plots(pipeline)
        }
    
    @staticmethod
    def get_related_pipelines(pipeline: Pipeline) -> List[Pipeline]:
        """Get pipelines related to the given pipeline (same dataset).
        
        Args:
            pipeline: The pipeline to find related pipelines for
            
        Returns:
            List of related pipeline objects
        """
        if not pipeline.dataset:
            return []
            
        return list(Pipeline.objects.filter(
            dataset=pipeline.dataset
        ).exclude(id=pipeline.pk).order_by('-created_at'))


class DatasetService:
    """Service for dataset operations."""
    
    @staticmethod
    def create_dataset(
        name: str, target_variable: str, selected_features: List[str] = None
    ) -> Dataset:
        """Create a new dataset.
        
        Args:
            name: Name for the dataset
            target_variable: Target variable name
            selected_features: List of features selected by the user
            
        Returns:
            The created Dataset object
        """
        return Dataset.objects.create(
            name=name,
            target_variable=target_variable,
            user_selected_features=selected_features or []
        )
    
    @staticmethod
    def update_selected_features(
        dataset: Dataset, selected_features: List[str]
    ) -> Dataset:
        """Update the selected features for a dataset.
        
        Args:
            dataset: The dataset to update
            selected_features: List of features selected by the user
            
        Returns:
            The updated Dataset object
        """
        dataset.user_selected_features = selected_features
        dataset.save() 