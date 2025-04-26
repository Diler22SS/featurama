"""Data models for the Featurama application.

This module defines the database models used by the application for storing
datasets, pipelines, and their relationships.
"""

from django.db import models
import os


# Create your models here.
class Dataset(models.Model):
    """Dataset model for storing data files and metadata.
    
    A Dataset represents an uploaded data file along with metadata about it,
    such as the target variable and selected features.
    
    Attributes:
        name: Human-readable name for the dataset
        file: File field for the actual dataset file
        uploaded_at: When the dataset was uploaded
        target_variable: Name of the target variable/column
        user_selected_features: JSON field storing features selected by users
    """
    
    name = models.CharField(max_length=255, null=True, blank=True)
    file = models.FileField(upload_to='datasets/', null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    target_variable = models.CharField(max_length=255, null=True, blank=True)
    user_selected_features = models.JSONField(null=True, blank=True)

    def __str__(self) -> str:
        """Return a string representation of the dataset."""
        return self.name or f"Dataset #{self.pk}"

    def get_filename(self) -> str:
        """Return just the filename without path."""
        if not self.file:
            return "No file"
        return os.path.basename(self.file.name)
    
    def get_file_extension(self) -> str:
        """Return the file extension."""
        if not self.file:
            return ""
        _, ext = os.path.splitext(self.file.name)
        return ext.lower()
    
    def is_csv(self) -> bool:
        """Check if the file is a CSV."""
        return self.get_file_extension() == '.csv'


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

    class Meta:
        """Meta options for the Pipeline model."""
        ordering = ['-created_at']
        verbose_name = "Feature Selection Pipeline"
        verbose_name_plural = "Feature Selection Pipelines"

    def __str__(self) -> str:
        """Return a string representation of the pipeline."""
        dataset_name = self.dataset.name if self.dataset else "No dataset"
        return f"Pipeline #{self.pk} for {dataset_name}"
    
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
            return "Awaiting dataset upload"
        if not self.is_configured():
            return "Awaiting configuration"
        return "Ready for execution"
