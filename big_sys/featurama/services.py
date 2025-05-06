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
            'Variance Threshold': 'Variance Threshold – удаляет признаки с низкой дисперсией',
            'ANOVA': 'ANOVA – выбирает признаки на основе F-значения',
            'Mutual Information': 'Mutual Information – выбирает признаки на основе взаимной информации',
            'MRMR': 'MRMR – Minimum Redundancy Maximum Relevance'
        }
        
        wrapper_methods = {
            'RFE with Logistic Regression': 'Recursive Feature Elimination с логистической регрессией',
            'RFE with Decision Tree': 'Recursive Feature Elimination с деревом решений',
            'SFS with Logistic Regression': 'Sequential Feature Selection с логистической регрессией',
            'SFS with Decision Tree': 'Sequential Feature Selection с деревом решений'
        }
        
        model_methods = {
            'Logistic Regression': 'Logistic Regression – линейная модель для классификации',
            'XGBoost Linear': 'XGBoost – градиентный бустинг с линейным бустером',
            'Decision Tree': 'Decision Tree – простая модель дерева решений',
            'XGBoost Tree': 'XGBoost – градиентный бустинг с деревом решений'
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
            
            if result and result.manual_features:
                return result.manual_features
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
                
                distribution_url = None
                if shap.distribution_explanation_image:
                    distribution_url = shap.distribution_explanation_image.url
                
                return {
                    'global': global_url,
                    'distribution': distribution_url
                }
        except ShapExplanation.DoesNotExist:
            pass
        
        return {
            'global': None,
            'distribution': None
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