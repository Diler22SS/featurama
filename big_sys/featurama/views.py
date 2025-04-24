from django.shortcuts import render
from django.http import HttpRequest, HttpResponse


# Create your views here.
def main(request: HttpRequest) -> HttpResponse:
    return HttpResponse('Hello world')
