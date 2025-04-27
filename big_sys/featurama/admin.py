from django.contrib import admin
from .models import Pipeline, Dataset, FeatureSelectionResult


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'name', 'uploaded_at', 'target_variable', 
        'user_selected_features'
    )
    list_filter = ('uploaded_at',)
    search_fields = ('id', 'name', 'target_variable')
    date_hierarchy = 'uploaded_at'


@admin.register(Pipeline)
class PipelineAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'dataset', 'created_at', 'filter_method',
        'wrapper_method', 'model_method'
    )
    list_filter = (
        'created_at', 'filter_method', 'wrapper_method', 
        'model_method'
    )
    search_fields = ('id', 'dataset__name')
    date_hierarchy = 'created_at'


@admin.register(FeatureSelectionResult)
class FeatureSelectionResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'pipeline', 'created_at', 'selected_features')
    list_filter = ('created_at',)
    search_fields = ('id', 'pipeline__id')
    date_hierarchy = 'created_at'
