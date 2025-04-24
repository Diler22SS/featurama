from django.urls import path
from . import views
from django.views.generic import RedirectView

app_name = 'featurama'
urlpatterns = [
    path('',
         RedirectView.as_view(url='featurama/pipelines', permanent=False)),
    path('featurama/pipelines',
         views.pipelines, name='pipelines'),
    path('featurama/pipelines/<int:pipeline_id>/upload_data',
         views.upload_data, name='upload_data'),
    path('featurama/pipelines/<int:pipeline_id>/delete',
         views.delete_pipeline, name='delete_pipeline'),
]
