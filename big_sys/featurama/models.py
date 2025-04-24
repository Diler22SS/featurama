from django.db import models


# Create your models here.
class Dataset(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    file = models.FileField(upload_to='datasets/', null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    target_variable = models.CharField(max_length=255, null=True, blank=True)
    user_selected_features = models.JSONField(null=True, blank=True)

    def __str__(self):
        return self.name or f"Dataset #{self.id}"


class Pipeline(models.Model):
    # name = models.CharField(max_length=255) Name is not neccessary yet
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='pipelines',
        null=True,  # In case a pipeline is created before linking to a dataset
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    filter_method = models.CharField(max_length=100, null=True, blank=True)
    wrapper_method = models.CharField(max_length=100, null=True, blank=True)
    model_method = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        dataset_name = self.dataset.name if self.dataset else "No dataset"
        return f"Pipeline #{self.id} for {dataset_name}"
