"""Data models for the Featurama application.

This module defines the database models used by the application for storing
datasets, pipelines, and their relationships.
"""

from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver
import os
import pandas as pd
from io import StringIO


# Create your models here.
class Dataset(models.Model):
    """Dataset model for storing data files and metadata.
    
    A Dataset represents an uploaded data file along with metadata about it,
    such as the target variable and selected features.
    
    Attributes:
        name: Human-readable name for the dataset
        data: JSON field storing the actual dataset data
        uploaded_at: When the dataset was uploaded
        target_variable: Name of the target variable/column
        user_selected_features: JSON field storing features selected by users
    """
    
    name = models.CharField(max_length=255, null=True, blank=True)
    data = models.JSONField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    target_variable = models.CharField(max_length=255, null=True, blank=True)
    user_selected_features = models.JSONField(null=True, blank=True)

    def __str__(self) -> str:
        """Return a string representation of the dataset."""
        return str(self.name or f"Набор данных #{self.pk}")

    def get_dataframe(self) -> pd.DataFrame:
        """Return the data as a pandas DataFrame."""
        if not self.data:
            return pd.DataFrame()
        return pd.read_json(StringIO(self.data))
    
    def save_dataframe(self, df: pd.DataFrame) -> None:
        """Save a pandas DataFrame as JSON."""
        self.data = df.to_json(orient='records')
        self.save()
    
    def get_filename(self) -> str:
        """Return just the filename without path."""
        if not self.data:
            return "Нет файла"
        return self.name
    
    def get_file_extension(self) -> str:
        """Return the file extension."""
        return ""
    
    def is_csv(self) -> bool:
        """Check if the file is a CSV."""
        return False
    
    def delete(self, *args, **kwargs):
        """Override delete to remove the associated file."""
        # Call the "real" delete() method
        super().delete(*args, **kwargs)


class Pipeline(models.Model):
    """Pipeline model for feature selection workflows.
    
    A Pipeline represents a complete feature selection workflow, from dataset
    to selected methods and results.
    
    Attributes:
        dataset: Foreign key to the associated Dataset
        created_at: When the pipeline was created
        filter_method: The selected filter method
        wrapper_method: The selected wrapper method
        model_method: The selected model method
        filter_params: JSON field storing parameters for the filter method
        wrapper_params: JSON field storing parameters for the wrapper method
        model_params: JSON field storing parameters for the model method
    """
    
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='pipelines',
        null=True,
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    filter_method = models.CharField(max_length=100, null=True, blank=True)
    wrapper_method = models.CharField(max_length=100, null=True, blank=True)
    model_method = models.CharField(max_length=100, null=True, blank=True)
    filter_params = models.JSONField(null=True, blank=True)
    wrapper_params = models.JSONField(null=True, blank=True)
    model_params = models.JSONField(null=True, blank=True)

    class Meta:
        """Meta options for the Pipeline model."""
        ordering = ['-created_at']
        verbose_name = "Пайплайн выбора признаков"
        verbose_name_plural = "Пайплайны выбора признаков"

    def __str__(self) -> str:
        """Return a string representation of the pipeline."""
        dataset_name = self.dataset.name if self.dataset else "Нет набора данных"
        return f"Пайплайн #{self.pk} для {dataset_name}"
    
    def is_dataset_uploaded(self) -> bool:
        """Check if a dataset has been uploaded to this pipeline."""
        return self.dataset is not None
    
    def is_configured(self) -> bool:
        """Check if all pipeline methods have been configured."""
        return all([
            self.filter_method, 
            self.wrapper_method, 
            self.model_method
        ])
    
    def get_configuration_status(self) -> str:
        """Get a human-readable status of the pipeline configuration."""
        if not self.is_dataset_uploaded():
            return "Ожидает загрузки набора данных"
        if not self.is_configured():
            return "Ожидает настройки"
        return "Готов к выполнению"


class FeatureSelectionResult(models.Model):
    """Feature selection result model.
    
    Stores the results of a feature selection process for a pipeline.
    
    Attributes:
        pipeline: Foreign key to the associated Pipeline
        filtered_features: JSON field storing features selected by the 
            algorithm after the filter method
        wrapped_features: JSON field storing features selected by the 
            algorithm after the wrapper method
        manual_features: JSON field storing features selected by the 
            user manually
        created_at: When the result was created
    """
    
    pipeline = models.ForeignKey(
        Pipeline,
        on_delete=models.CASCADE,
        related_name='feature_selection_results'
    )
    filtered_features = models.JSONField(null=True, blank=True)
    wrapped_features = models.JSONField(null=True, blank=True)
    manual_features = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        """Meta options for the FeatureSelectionResult model."""
        ordering = ['-created_at']
        verbose_name = "Результат выбора признаков"
        verbose_name_plural = "Результаты выбора признаков"
    
    def __str__(self) -> str:
        """Return a string representation of the result."""
        return f"Результат выбора для пайплайна #{self.pipeline.id}"


class PerformanceMetric(models.Model):
    """Performance metrics model.
    
    Stores evaluation metrics for a pipeline's machine learning model.
    
    Attributes:
        pipeline: Foreign key to the associated Pipeline
        roc_auc: ROC AUC score of the model
        accuracy: Accuracy score of the model
        f1_score: F1 score of the model
        created_at: When the metrics were created
    """
    
    pipeline = models.ForeignKey(
        Pipeline,
        on_delete=models.CASCADE,
        related_name='performance_metrics'
    )
    roc_auc = models.FloatField(null=True, blank=True)
    accuracy = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        """Meta options for the PerformanceMetric model."""
        ordering = ['-created_at']
        verbose_name = "Метрика производительности"
        verbose_name_plural = "Метрики производительности"
    
    def __str__(self) -> str:
        """Return a string representation of the metrics."""
        return f"Метрики для пайплайна #{self.pipeline.id}"


class ShapExplanation(models.Model):
    """SHAP explanation model.
    
    Stores SHAP explanation visuals for a pipeline's model.
    
    Attributes:
        pipeline: Foreign key to the associated Pipeline
        global_explanation_image: Image for global feature importance
        distribution_explanation_image: Image for distribution feature importance
        created_at: When the explanation was created
    """
    
    pipeline = models.ForeignKey(
        Pipeline,
        on_delete=models.CASCADE,
        related_name='shap_explanations'
    )
    global_explanation_image = models.ImageField(
        upload_to='shap_explanations/',
        null=True,
        blank=True
    )
    distribution_explanation_image = models.ImageField(
        upload_to='shap_explanations/',
        null=True,
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        """Meta options for the ShapExplanation model."""
        ordering = ['-created_at']
        verbose_name = "SHAP объяснение"
        verbose_name_plural = "SHAP объяснения"
    
    def __str__(self) -> str:
        """Return a string representation of the explanation."""
        return f"SHAP объяснение для пайплайна #{self.pipeline.id}"
    
    def delete(self, *args, **kwargs):
        """Override delete to remove the associated images."""
        # Delete the global explanation image if it exists
        if self.global_explanation_image:
            if os.path.isfile(self.global_explanation_image.path):
                os.remove(self.global_explanation_image.path)
        
        # Delete the distribution explanation image if it exists 
        if self.distribution_explanation_image:
            if os.path.isfile(self.distribution_explanation_image.path):
                os.remove(self.distribution_explanation_image.path)
        
        # Call the "real" delete() method
        super().delete(*args, **kwargs)


# Signal handlers to handle file deletion on model deletion
@receiver(pre_delete, sender=Dataset)
def delete_dataset_files(sender, instance, **kwargs):
    """Delete files when a Dataset is deleted."""
    if instance.data:
        # Delete the associated file
        if os.path.isfile(instance.data):
            os.remove(instance.data)


@receiver(pre_delete, sender=ShapExplanation)
def delete_shap_files(sender, instance, **kwargs):
    """Delete image files when a ShapExplanation is deleted."""
    # Delete global explanation image
    if instance.global_explanation_image:
        if os.path.isfile(instance.global_explanation_image.path):
            os.remove(instance.global_explanation_image.path)
    
    # Delete distribution explanation image
    if instance.distribution_explanation_image:
        if os.path.isfile(instance.distribution_explanation_image.path):
            os.remove(instance.distribution_explanation_image.path)
