"""Feature selection algorithms for the Featurama application.

This module contains algorithms used for feature selection in pipelines.
"""

# Standard library imports
import io
import random

# Third-party imports
import matplotlib
matplotlib.use('Agg')  # Must be called before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from django.core.files.base import ContentFile
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from pymrmr import mRMR
from sklearn.feature_selection import (
    RFE, SelectKBest, VarianceThreshold, 
    f_classif, mutual_info_classif
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Local application imports
from .models import (
    FeatureSelectionResult, PerformanceMetric, 
    Pipeline, ShapExplanation
)


# Filter Methods
def filter_variance_threshold(pipeline: Pipeline, X, y=None, **kwargs) -> FeatureSelectionResult:
    """Remove features with variance below a specified threshold."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return FeatureSelectionResult.objects.create(
            pipeline=pipeline,
            filtered_features=[]
        )

    all_features = pipeline.dataset.user_selected_features
    print(f"Filter selected features IN variance threshold: {all_features}")
    X_all = X[all_features]

    threshold = kwargs.get('threshold', 0.1)
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_all, y)
    selected_columns = X_all.columns[selector.get_support()]
    print(f"Filter selected features OUT variance threshold: {selected_columns}")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=selected_columns.tolist()
    )


def filter_anova(pipeline: Pipeline, X, y, **kwargs) -> FeatureSelectionResult:
    """Select top k features based on ANOVA F-value."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return FeatureSelectionResult.objects.create(
            pipeline=pipeline,
            filtered_features=[]
        )

    all_features = pipeline.dataset.user_selected_features
    print(f"Filter selected features IN anova: {all_features}")
    X_all = X[all_features]

    k = kwargs.get('k', 'all')
    # Scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns)
    
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_scaled, y)

    scores = selector.scores_
    p_values = selector.pvalues_
    feature_scores = pd.DataFrame({
        'feature': X_all.columns,
        'f_score': scores,
        'p_value': p_values
    })
    print(feature_scores)

    top_features = feature_scores[feature_scores['p_value'] < 0.05]['feature'].tolist()
    print(f"Filter selected features OUT anova: {top_features}")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=top_features
    )


def filter_mutual_info(pipeline: Pipeline, X, y, **kwargs) -> FeatureSelectionResult:
    """Select top k features based on mutual information."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return FeatureSelectionResult.objects.create(
            pipeline=pipeline,
            filtered_features=[]
        )

    all_features = pipeline.dataset.user_selected_features
    print(f"Filter selected features IN mutual info: {all_features}")
    X_all = X[all_features]

    k = kwargs.get('k', 'all')
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X_all, y)  
    scores = selector.scores_
    feature_scores = pd.DataFrame({
        'feature': X_all.columns,
        'mutual_info': scores,
    })
    print(feature_scores)

    top_features = feature_scores[feature_scores['mutual_info'] > 0.1]['feature'].tolist()
    print(f"Filter selected features OUT mutual info: {top_features}")
    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=top_features
    )


def filter_mrmr(pipeline: Pipeline, X, y, **kwargs) -> FeatureSelectionResult:
    """Select features using minimum Redundancy Maximum Relevance (mRMR)."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return FeatureSelectionResult.objects.create(
            pipeline=pipeline,
            filtered_features=[]
        )   

    all_features = pipeline.dataset.user_selected_features
    print(f"Filter selected features IN mRMR: {all_features}")
    X_all = X[all_features] 

    k = kwargs.get('k', 0.5 * X.shape[1])
    data = X_all.copy()
    data['target'] = y
    selected = mRMR(data, 'MIQ', k)
    # Remove 'target' from selected features if present
    selected = [col for col in selected if col != 'target']
    if not selected:
        raise ValueError("No valid features selected by mRMR")
    print(f"Filter selected features OUT mRMR: {selected}")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=selected
    )


# Wrapper Methods
def wrapper_sfs_logreg(pipeline: Pipeline, X, y, **kwargs) -> FeatureSelectionResult:
    """Perform forward feature selection with Logistic Regression."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return FeatureSelectionResult.objects.create(
            pipeline=pipeline,
            filtered_features=[],
            wrapped_features=[]
        )

    # Get filtered features from previous step if available
    filtered_features = FeatureSelectionResult.objects.filter(pipeline=pipeline).first().filtered_features
    print(f"Wrapper selected features IN sfs logreg: {filtered_features}")
    X_filtered = X[filtered_features]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered), columns=X_filtered.columns)
    
    model = LogisticRegression(
        penalty='l2',          # Use L2 regularization for stability
        C=1.0,                 # Moderate regularization strength
        solver='liblinear',    # Efficient solver for L2 penalty
        max_iter=1000,         # Ensure convergence for complex datasets
        random_state=42,
    )
    k_features = kwargs.get('k_features', 'best')
    sfs = SFS(model, k_features=k_features, forward=True, floating=False, scoring='roc_auc', verbose=2, cv=5)
    sfs.fit(X_scaled.values, y.values)
    selected_features = list(X_filtered.columns[list(sfs.k_feature_idx_)])
    print(f"Wrapper selected features OUT sfs logreg: {selected_features}")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=filtered_features,
        wrapped_features=selected_features
    )


def wrapper_sfs_tree(pipeline: Pipeline, X, y, **kwargs) -> FeatureSelectionResult:
    """Perform forward feature selection with Decision Tree."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return FeatureSelectionResult.objects.create(
            pipeline=pipeline,
            filtered_features=[],
            wrapped_features=[]
        )   

    # Get filtered features from previous step if available
    filtered_features = FeatureSelectionResult.objects.filter(pipeline=pipeline).first().filtered_features
    print(f"Wrapper selected features IN sfs tree: {filtered_features}")
    X_filtered = X[filtered_features]

    model = DecisionTreeClassifier(
        criterion='gini',           # Splitting criterion: 'gini' or 'entropy'
        splitter='best',            # Strategy for splitting: 'best' or 'random'
        max_depth=5,             # Maximum depth of the tree (None means unlimited)
        min_samples_split=10,        # Minimum number of samples required to split an internal node
        min_samples_leaf=5,         # Minimum number of samples required to be at a leaf node
        max_features=None,          # Number of features to consider when looking for the best split
        random_state=42             # For reproducibility
    )
    k_features = kwargs.get('k_features', 'best')
    sfs = SFS(model, k_features=k_features, forward=True, floating=False, scoring='roc_auc', verbose=2, cv=5)
    sfs.fit(X_filtered.values, y.values)
    selected_features = list(X_filtered.columns[list(sfs.k_feature_idx_)])
    print(f"Wrapper selected features OUT sfs tree: {selected_features}")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=filtered_features,
        wrapped_features=selected_features
    )


def wrapper_rfe_logreg(pipeline: Pipeline, X, y, **kwargs) -> FeatureSelectionResult:
    """Perform recursive feature elimination with Logistic Regression."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return FeatureSelectionResult.objects.create(
            pipeline=pipeline,
            filtered_features=[],
            wrapped_features=[]
        )

    # Get filtered features from previous step if available
    filtered_features = FeatureSelectionResult.objects.filter(pipeline=pipeline).first().filtered_features
    print(f"Wrapper selected features IN rfe logreg: {filtered_features}")
    X_filtered = X[filtered_features]

    n_features = kwargs.get('n_features_to_select', None)
    # Scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_filtered), columns=X_filtered.columns)
    
    model = LogisticRegression(
        penalty='l2',          # Use L2 regularization for stability
        C=1.0,                 # Moderate regularization strength
        solver='liblinear',    # Efficient solver for L2 penalty
        max_iter=1000,         # Ensure convergence for complex datasets
        random_state=42,
    )
    rfe = RFE(estimator=model, n_features_to_select=n_features, verbose=2)
    rfe.fit(X_scaled, y)
    selected_features = list(X_filtered.columns[rfe.support_])
    print(f"Wrapper selected features OUT rfe logreg: {selected_features}")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=filtered_features,
        wrapped_features=selected_features
    )


def wrapper_rfe_tree(pipeline: Pipeline, X, y, **kwargs) -> FeatureSelectionResult:
    """Perform recursive feature elimination with Decision Tree."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return FeatureSelectionResult.objects.create(
            pipeline=pipeline,
            filtered_features=[],
            wrapped_features=[]
        )

    # Get filtered features from previous step if available
    filtered_features = FeatureSelectionResult.objects.filter(pipeline=pipeline).first().filtered_features
    print(f"Wrapper selected features IN rfe tree: {filtered_features}")
    X_filtered = X[filtered_features]

    n_features = kwargs.get('n_features_to_select', None)
    model = DecisionTreeClassifier(
        criterion='gini',           # Splitting criterion: 'gini' or 'entropy'
        splitter='best',            # Strategy for splitting: 'best' or 'random'
        max_depth=5,             # Maximum depth of the tree (None means unlimited)
        min_samples_split=10,        # Minimum number of samples required to split an internal node
        min_samples_leaf=5,         # Minimum number of samples required to be at a leaf node
        max_features=None,          # Number of features to consider when looking for the best split
        random_state=42             # For reproducibility
    )
    rfe = RFE(estimator=model, n_features_to_select=n_features, verbose=2)
    rfe.fit(X_filtered, y)
    selected_features = list(X_filtered.columns[rfe.support_])
    print(f"Wrapper selected features OUT rfe tree: {selected_features}")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=filtered_features,
        wrapped_features=selected_features
    )


# SHAP Methods
def shap_linear_explainer(model, X, **kwargs) -> tuple:
    """Compute SHAP values using LinearExplainer."""
    explainer = shap.LinearExplainer(model, X, **kwargs)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def shap_tree_explainer( model, X, **kwargs) -> tuple:
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model, **kwargs)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def _generate_global_shap_plot(X, explainer, shap_values) -> bytes:
    """Generate a random global SHAP plot.
    
    Args:
        pipeline: The pipeline to generate the plot for
        
    Returns:
        The plot as bytes
    """
    # 1. Summary plot (глобальная важность признаков)
    plt.figure(figsize=(14, 6), dpi=150)  # Wider and higher resolution
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=X.columns, 
        plot_type="bar", 
        show=False
    )
    plt.xlabel("mean(|SHAP value|) (impact on model output magnitude)", fontsize=12)  # Adjust fontsize here
    plt.tick_params(axis='y', labelsize=10)
    plt.title("Важность признаков", fontsize=14)
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return buf.getvalue()


def _generate_local_shap_plot(pipeline: Pipeline) -> bytes:
    """Generate a random local SHAP plot.
    
    Args:
        pipeline: The pipeline to generate the plot for
        
    Returns:
        The plot as bytes
    """
    plt.figure(figsize=(10, 6))
    
    # Get feature names from the pipeline
    features = []
    if pipeline.dataset and pipeline.dataset.user_selected_features:
        # Take fewer features for local explanation (max 5)
        features = pipeline.dataset.user_selected_features[:5]
    else:
        # Use generic feature names if no real features available
        features = [f'Feature {i}' for i in range(5)]
    
    # Generate random SHAP values (both positive and negative)
    shap_values = [random.uniform(-1, 1) for _ in range(len(features))]
    
    # Create colors based on value sign
    colors = ['red' if val < 0 else 'blue' for val in shap_values]
    
    # Create horizontal bar plot
    plt.barh(features, shap_values, color=colors)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel('SHAP Value')
    plt.title('Local Feature Impact (Sample Prediction)')
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return buf.getvalue()


# Model Methods
def model_logistic_regression(pipeline: Pipeline, X, y, **kwargs) -> tuple:
    """Train and evaluate a Logistic Regression model."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return PerformanceMetric.objects.create(
            pipeline=pipeline,
            roc_auc=0.0,
            accuracy=0.0,
            f1_score=0.0
        ), ShapExplanation.objects.create(
            pipeline=pipeline
        )

    # Get wrapped features from previous step if available
    wrapped_features = FeatureSelectionResult.objects.filter(pipeline=pipeline).first().wrapped_features
    print(f"Model selected features IN logreg: {wrapped_features}")
    X_wrapped = X[wrapped_features]

    X_train, X_test, y_train, y_test = train_test_split(X_wrapped, y, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    model = LogisticRegression(
        penalty='l2',          # Use L2 regularization for stability
        C=1.0,                 # Moderate regularization strength
        solver='liblinear',    # Efficient solver for L2 penalty
        max_iter=1000,         # Ensure convergence for complex datasets
        random_state=42,
        **kwargs
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    explainer, shap_values = shap_linear_explainer(model, X_test_scaled)

    # Generate and save global and local explanation plot
    global_plot = _generate_global_shap_plot(X_test_scaled, explainer, shap_values)
    local_plot = _generate_local_shap_plot(pipeline) # TODO: add local plot

    metrics = {
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    print(f"Model metrics OUT logreg: {metrics}")

    return PerformanceMetric.objects.create(
        pipeline=pipeline,
        **metrics
    ), ShapExplanation.objects.create(
        pipeline=pipeline,
        global_explanation_image=ContentFile(global_plot, name=f'global_shap_{pipeline.id}.png'),
        local_explanation_image=ContentFile(local_plot, name=f'local_shap_{pipeline.id}.png')
    )


def model_xgb_linear(pipeline: Pipeline, X, y, **kwargs) -> tuple:
    """Train and evaluate an XGBoost model with linear booster."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return PerformanceMetric.objects.create(
            pipeline=pipeline,
            roc_auc=0.0,
            accuracy=0.0,
            f1_score=0.0
        ), ShapExplanation.objects.create(
            pipeline=pipeline
        )

    # Get wrapped features from previous step if available
    wrapped_features = FeatureSelectionResult.objects.filter(pipeline=pipeline).first().wrapped_features
    print(f"Model selected features IN xgb linear: {wrapped_features}")
    X_wrapped = X[wrapped_features]

    X_train, X_test, y_train, y_test = train_test_split(X_wrapped, y, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    model = XGBClassifier(booster='gblinear', eval_metric='auc', random_state=42, **kwargs)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    explainer, shap_values = shap_linear_explainer(model, X_test_scaled)

    # Generate and save global and local explanation plot
    global_plot = _generate_global_shap_plot(X_test_scaled, explainer, shap_values)
    local_plot = _generate_local_shap_plot(pipeline) # TODO: add local plot

    metrics = {
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    print(f"Model metrics OUT xgb linear: {metrics}")

    return PerformanceMetric.objects.create(
        pipeline=pipeline,
        **metrics
    ), ShapExplanation.objects.create(
        pipeline=pipeline,
        global_explanation_image=ContentFile(global_plot, name=f'global_shap_{pipeline.id}.png'),
        local_explanation_image=ContentFile(local_plot, name=f'local_shap_{pipeline.id}.png')
    )


def model_decision_tree(pipeline: Pipeline, X, y, **kwargs) -> tuple:
    """Train and evaluate a Decision Tree model."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return PerformanceMetric.objects.create(
            pipeline=pipeline,
            roc_auc=0.0,
            accuracy=0.0,
            f1_score=0.0
        ), ShapExplanation.objects.create(
            pipeline=pipeline
        )

    # Get wrapped features from previous step if available
    wrapped_features = FeatureSelectionResult.objects.filter(pipeline=pipeline).first().wrapped_features
    print(f"Model selected features IN decision tree: {wrapped_features}")
    X_wrapped = X[wrapped_features]

    X_train, X_test, y_train, y_test = train_test_split(X_wrapped, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(
        criterion='gini',           # Splitting criterion: 'gini' or 'entropy'
        splitter='best',            # Strategy for splitting: 'best' or 'random'
        max_depth=5,                # Maximum depth of the tree (None means unlimited)
        min_samples_split=10,       # Minimum number of samples required to split an internal node
        min_samples_leaf=5,         # Minimum number of samples required to be at a leaf node
        max_features=None,          # Number of features to consider when looking for the best split
        random_state=42,            # For reproducibility
        **kwargs
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    explainer, shap_values = shap_tree_explainer(model, X_test)

    # Generate and save global and local explanation plot
    global_plot = _generate_global_shap_plot(X_test, explainer, shap_values)
    local_plot = _generate_local_shap_plot(pipeline) # TODO: add local plot

    metrics = {
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    print(f"Model metrics OUT decision tree: {metrics}")

    return PerformanceMetric.objects.create(
        pipeline=pipeline,
        **metrics
    ), ShapExplanation.objects.create(
        pipeline=pipeline,
        global_explanation_image=ContentFile(global_plot, name=f'global_shap_{pipeline.id}.png'),
        local_explanation_image=ContentFile(local_plot, name=f'local_shap_{pipeline.id}.png')
    )


def model_xgb_tree(pipeline: Pipeline, X, y, **kwargs) -> tuple:
    """Train and evaluate an XGBoost model with tree booster."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        # If no features are available, return empty result
        return PerformanceMetric.objects.create(
            pipeline=pipeline,
            roc_auc=0.0,
            accuracy=0.0,
            f1_score=0.0
        ), ShapExplanation.objects.create(
            pipeline=pipeline
        )

    # Get wrapped features from previous step if available
    wrapped_features = FeatureSelectionResult.objects.filter(pipeline=pipeline).first().wrapped_features
    print(f"Model selected features IN xgb tree: {wrapped_features}")
    X_wrapped = X[wrapped_features]

    X_train, X_test, y_train, y_test = train_test_split(X_wrapped, y, test_size=0.2, random_state=42)

    model = XGBClassifier(booster='gbtree', eval_metric='auc', random_state=42, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    explainer, shap_values = shap_tree_explainer(model, X_test)

    # Generate and save global and local explanation plot
    global_plot = _generate_global_shap_plot(X_test, explainer, shap_values)
    local_plot = _generate_local_shap_plot(pipeline) # TODO: add local plot

    metrics = {
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    print(f"Model metrics OUT xgb tree: {metrics}")

    return PerformanceMetric.objects.create(
        pipeline=pipeline,
        **metrics
    ), ShapExplanation.objects.create(
        pipeline=pipeline,
        global_explanation_image=ContentFile(global_plot, name=f'global_shap_{pipeline.id}.png'),
        local_explanation_image=ContentFile(local_plot, name=f'local_shap_{pipeline.id}.png')
    )


# Method Dictionaries
FILTER_METHODS = {
    'variance_threshold': filter_variance_threshold,
    'anova': filter_anova,
    'mutual_info': filter_mutual_info,
    'mrmr': filter_mrmr,
}

WRAPPER_METHODS = {
    'sfs_logreg': wrapper_sfs_logreg,
    'sfs_tree': wrapper_sfs_tree,
    'rfe_logreg': wrapper_rfe_logreg,
    'rfe_tree': wrapper_rfe_tree,
}

MODEL_METHODS = {
    'logreg': model_logistic_regression,
    'xgb_linear': model_xgb_linear,
    'dtree': model_decision_tree,
    'xgb_tree': model_xgb_tree,
}


def run_pipeline(pipeline: Pipeline) -> None:
    """Run all pipeline analysis steps.
    
    This function runs feature selection, model evaluation, and SHAP analysis
    for a pipeline. It's a comprehensive function that can be called after
    pipeline configuration is complete.
    
    Args:
        pipeline: The pipeline to analyze
    """
    # Get the dataset
    df = pipeline.dataset.get_dataframe()
    target_var = pipeline.dataset.target_variable
    
    # Split into features and target
    X = df.drop(columns=[target_var])
    y = df[target_var]

    config = {
        'filter_method': pipeline.filter_method,
        'wrapper_method': pipeline.wrapper_method,
        'model_method': pipeline.model_method,
    }
    print(f"Running pipeline with config: {config}")

    # Validate configuration
    required_keys = ['model_method']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")  

    if config.get('filter_method') and config['filter_method'] not in FILTER_METHODS:
        raise ValueError(f"Invalid filter_method: {config['filter_method']}")
    if config.get('wrapper_method') and config['wrapper_method'] not in WRAPPER_METHODS:
        raise ValueError(f"Invalid wrapper_method: {config['wrapper_method']}")
    if config['model_method'] not in MODEL_METHODS:
        raise ValueError(f"Invalid model_method: {config['model_method']}")

    # Step 1: Filter Method (Optional)
    if config.get('filter_method'):
        filter_func = FILTER_METHODS[config['filter_method']]
        filter_params = config.get('filter_params', {})
        filter_func(pipeline, X.copy(), y.copy(), **filter_params)

    # Step 2: Wrapper Method (Optional)
    if config.get('wrapper_method'):
        wrapper_func = WRAPPER_METHODS[config['wrapper_method']]
        wrapper_params = config.get('wrapper_params', {})
        wrapper_func(pipeline, X.copy(), y.copy(), **wrapper_params)

    # Step 3: Model Training and SHAP Analysis
    model_func = MODEL_METHODS[config['model_method']]
    model_params = config.get('model_params', {})
    model_func(pipeline, X.copy(), y.copy(), **model_params)
