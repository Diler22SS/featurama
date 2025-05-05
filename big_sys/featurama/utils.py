"""Utility functions for the Featurama application."""

import os
import tempfile
from typing import Tuple, Optional, List

import pandas as pd
from django.http import HttpRequest


def read_dataset_file(file) -> Tuple[pd.DataFrame, str]:
    """Read a dataset file and return a dataframe with its temp path.
    
    Args:
        file: The uploaded file object
        
    Returns:
        Tuple of (dataframe, temp_file_path)
    """
    # Store the file temporarily first
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    for chunk in file.chunks():
        temp_file.write(chunk)
    temp_file.close()
    
    # Now read the file based on extension
    if file.name.endswith('.csv'):
        df = pd.read_csv(temp_file.name)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(temp_file.name, engine='openpyxl')
    elif file.name.endswith('.xls'):
        df = pd.read_excel(temp_file.name, engine='xlrd')
    else:
        # Default to CSV for unknown types
        df = pd.read_csv(temp_file.name)
    
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


def validate_dataset(
    df: pd.DataFrame, 
    target_variable: str, 
    selected_features: List[str]
) -> Tuple[bool, Optional[str]]:
    """Validate dataset against requirements.
    
    Args:
        df: The pandas DataFrame to validate
        target_variable: The name of the target variable column
        selected_features: List of feature columns to include in validation
        
    Returns:
        Tuple of (is_valid, error_message)
        is_valid: True if dataset passes all checks, False otherwise
        error_message: None if valid, otherwise error explanation
    """
    # Validate that the target variable exists
    if target_variable not in df.columns:
        msg = f"Target variable '{target_variable}' not found in dataset"
        return False, msg
    
    # Subset the dataframe to only include selected features and target
    columns_to_check = selected_features + [target_variable]
    df_subset = df[columns_to_check]
    
    # 1. Check for missing values
    missing_check, missing_error = check_no_missing_values(df_subset)
    if not missing_check:
        return False, missing_error
    
    # 2. Check that target variable is binary
    binary_check, binary_error = check_binary_target(df, target_variable)
    if not binary_check:
        return False, binary_error
    
    # 3. Check target variable balance
    balance_check, balance_error = check_target_balance(df, target_variable)
    if not balance_check:
        return False, balance_error
    
    # All checks passed
    return True, None


def check_no_missing_values(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Check that there are no missing values in the dataset.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_counts = df.isna().sum()
    columns_with_missing = missing_counts[missing_counts > 0]
    
    if len(columns_with_missing) > 0:
        # Format error message with columns and their missing counts
        column_details = ', '.join([
            f"{col}: {count} missing" 
            for col, count in columns_with_missing.items()
        ])
        return False, f"Dataset contains missing values: {column_details}"
    
    return True, None


def check_binary_target(
    df: pd.DataFrame, target_variable: str
) -> Tuple[bool, Optional[str]]:
    """Check that the target variable is binary (0/1).
    
    Args:
        df: DataFrame to check
        target_variable: Name of the target column
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Get unique values
    unique_values = df[target_variable].unique()
    
    # Check if values are only 0 and 1
    if set(unique_values) == {0, 1}:
        return True, None
    
    # Check if values are convertible to 0/1
    try:
        # Try to convert to numeric (if strings)
        numeric_values = pd.to_numeric(df[target_variable])
        unique_numeric = set(numeric_values.unique())
        
        if unique_numeric == {0, 1}:
            return True, None
        elif len(unique_numeric) == 2:
            error_msg = (
                f"Target variable '{target_variable}' has values "
                f"{unique_numeric} instead of required 0/1. "
                f"Consider encoding it before upload."
            )
            return False, error_msg
        else:
            error_msg = (
                f"Target variable '{target_variable}' has "
                f"{len(unique_numeric)} unique values instead of 2."
            )
            return False, error_msg
    except Exception:
        error_msg = (
            f"Target variable '{target_variable}' must be binary "
            f"(0/1). Found values: {unique_values[:5]}..."
        )
        return False, error_msg


def check_target_balance(
    df: pd.DataFrame, target_variable: str, threshold: float = 0.2
) -> Tuple[bool, Optional[str]]:
    """Check that the target variable is reasonably balanced.
    
    Args:
        df: DataFrame to check
        target_variable: Name of the target column
        threshold: Minimum acceptable proportion for minority class
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Convert to numeric if needed
        target_values = pd.to_numeric(df[target_variable])
        
        # Count occurrences of each class
        value_counts = target_values.value_counts(normalize=True)
        
        # Check if minority class is below threshold
        min_proportion = value_counts.min()
        
        if min_proportion < threshold:
            minority_class = value_counts.idxmin()
            error_msg = (
                f"Target variable '{target_variable}' is imbalanced. "
                f"Class {minority_class} represents only "
                f"{min_proportion:.1%} of the data. "
                f"Minimum required is {threshold:.1%}."
            )
            return False, error_msg
        
        return True, None
    except Exception:
        return False, f"Error checking target balance for '{target_variable}'" 