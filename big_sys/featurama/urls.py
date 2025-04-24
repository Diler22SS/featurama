from django.urls import path
from . import views
from django.views.generic import RedirectView

app_name = 'featurama'
urlpatterns = [
    path('', RedirectView.as_view(url='featurama/main', permanent=False)),
    path('featurama/main', views.main, name='main'),
]
