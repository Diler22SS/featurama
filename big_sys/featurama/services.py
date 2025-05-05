"""Service layer for Featurama business logic.

This module contains classes that handle the business logic for the application,
separating it from the view functions and keeping controllers thin.
"""

from typing import Any, Dict, List, Optional

from .models import (
    Dataset, FeatureSelectionResult, 
    PerformanceMetric, Pipeline, ShapExplanation
)


class MethodsService:
    """Service for managing pipeline methods and their descriptions."""
    
    @staticmethod
    def get_available_methods() -> Dict[str, Dict[str, str]]:
        """Get all available pipeline methods with descriptions."""
        filter_methods = {
            'variance_threshold': 'Variance Threshold – removes low-variance features',
            'anova': 'ANOVA – selects features based on F-value',
            'mutual_info': 'Mutual Information – selects features based on mutual info',
            'mrmr': 'MRMR – Minimum Redundancy Maximum Relevance'
        }
        
        wrapper_methods = {
            'rfe_logreg': 'Recursive Feature Elimination with Logistic Regression',
            'rfe_tree': 'Recursive Feature Elimination with Decision Tree',
            'sfs_logreg': 'Sequential Feature Selection with Logistic Regression',
            'sfs_tree': 'Sequential Feature Selection with Decision Tree'
        }
        
        model_methods = {
            'logreg': 'Logistic Regression – linear model for classification',
            'xgb_linear': 'XGBoost – gradient boosting with linear booster',
            'dtree': 'Decision Tree – simple decision tree model',
            'xgb_tree': 'XGBoost – gradient boosting with tree booster'
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
        """Get performance metrics for a pipeline."""
        try:
            metric = PerformanceMetric.objects.filter(
                pipeline=pipeline
            ).order_by('-created_at').first()
            
            if metric:
                return {
                    'roc_auc': metric.roc_auc,
                    'accuracy': metric.accuracy,
                    'f1': metric.f1_score
                }
        except PerformanceMetric.DoesNotExist:
            pass
            
        return {
            'roc_auc': None,
            'accuracy': None,
            'f1': None
        }
    
    @staticmethod
    def get_selected_features(pipeline: Pipeline) -> List[str]:
        """Get algorithm-selected features for a pipeline."""
        try:
            result = FeatureSelectionResult.objects.filter(
                pipeline=pipeline
            ).order_by('-created_at').first()
            
            if result and result.wrapped_features:
                return result.wrapped_features
        except FeatureSelectionResult.DoesNotExist:
            pass
        
        # Fall back to user selected features if algorithm hasn't run
        if pipeline.dataset and pipeline.dataset.user_selected_features:
            return pipeline.dataset.user_selected_features
        
        return []
    
    @staticmethod
    def get_user_selected_features(pipeline: Pipeline) -> List[str]:
        """Get user-selected features for a pipeline."""
        if pipeline.dataset and pipeline.dataset.user_selected_features:
            return pipeline.dataset.user_selected_features
        return []
    
    @staticmethod
    def get_shap_plots(pipeline: Pipeline) -> Dict[str, Optional[str]]:
        """Get SHAP plot URLs for a pipeline."""
        try:
            shap = ShapExplanation.objects.filter(
                pipeline=pipeline
            ).order_by('-created_at').first()
            
            if shap:
                global_url = None
                if shap.global_explanation_image:
                    global_url = shap.global_explanation_image.url
                
                local_url = None
                if shap.local_explanation_image:
                    local_url = shap.local_explanation_image.url
                
                return {
                    'global': global_url,
                    'local': local_url
                }
        except ShapExplanation.DoesNotExist:
            pass
        
        return {
            'global': None,
            'local': None
        }
    
    @classmethod
    def get_pipeline_results_context(cls, pipeline: Pipeline) -> Dict[str, Any]:
        """
        Get all results data for a pipeline in a format ready for templates.
        
        This method aggregates all results data for a pipeline for easy use
        in views.
        """
        return {
            'metrics': cls.get_pipeline_metrics(pipeline),
            'selected_features': cls.get_selected_features(pipeline),
            'user_selected_features': cls.get_user_selected_features(pipeline),
            'shap_plots': cls.get_shap_plots(pipeline)
        }
    
    @staticmethod
    def get_related_pipelines(pipeline: Pipeline) -> List[Pipeline]:
        """Get pipelines related to the given pipeline (same dataset)."""
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
        """Create a new dataset."""
        return Dataset.objects.create(
            name=name,
            target_variable=target_variable,
            user_selected_features=selected_features or []
        )
    
    @staticmethod
    def update_selected_features(
        dataset: Dataset, selected_features: List[str]
    ) -> Dataset:
        """Update the selected features for a dataset."""
        dataset.user_selected_features = selected_features
        dataset.save()
        return dataset 