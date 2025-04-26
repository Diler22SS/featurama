"""Utility functions for the Featurama application."""

import os
import tempfile
import pandas as pd
from typing import Tuple, Dict, List
from django.http import HttpRequest


def read_dataset_file(file) -> Tuple[pd.DataFrame, str]:
    """Read a dataset file and return a dataframe with its temp path.
    
    Args:
        file: The uploaded file object
        
    Returns:
        Tuple of (dataframe, temp_file_path)
    """
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:  # Excel file
        df = pd.read_excel(file)
    
    # Store the file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    for chunk in file.chunks():
        temp_file.write(chunk)
    temp_file.close()
    
    return df, temp_file.name


def clean_up_temp_file(request: HttpRequest) -> None:
    """Clean up temporary file and session data.
    
    Args:
        request: The HTTP request containing session data
    """
    temp_file_path = request.session.get('temp_file_path')
    if temp_file_path and os.path.exists(temp_file_path):
        os.unlink(temp_file_path)
    
    # Clear session data
    if 'temp_file_path' in request.session:
        del request.session['temp_file_path']
    if 'temp_file_name' in request.session:
        del request.session['temp_file_name'] 