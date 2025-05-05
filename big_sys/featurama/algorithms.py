"""Feature selection algorithms for the Featurama application.

This module contains algorithms used for feature selection in pipelines.
"""

# Standard library imports
import io
import random
from typing import Optional, Tuple, List, Dict, Any

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

# Constants
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.1
DEFAULT_K_FEATURES = 'best'
DEFAULT_CV = 5

class FeatureSelectionError(Exception):
    """Custom exception for feature selection errors."""
    pass

def _create_empty_result(pipeline: Pipeline) -> FeatureSelectionResult:
        """Create an empty feature selection result."""
        return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
            filtered_features=[],
            wrapped_features=[]
        )

def _create_empty_metrics(pipeline: Pipeline) -> Tuple[PerformanceMetric, ShapExplanation]:
    """Create empty performance metrics and SHAP explanation."""
    return (
        PerformanceMetric.objects.create(
            pipeline=pipeline,
            roc_auc=0.0,
            accuracy=0.0,
            f1_score=0.0
        ),
        ShapExplanation.objects.create(pipeline=pipeline)
    )

def _get_filtered_features(pipeline: Pipeline) -> List[str]:
    """Get filtered features from previous step if available."""
    result = FeatureSelectionResult.objects.filter(pipeline=pipeline).first()
    return result.filtered_features if result else []

def _get_wrapped_features(pipeline: Pipeline) -> List[str]:
    """Get wrapped features from previous step if available."""
    result = FeatureSelectionResult.objects.filter(pipeline=pipeline).first()
    return result.wrapped_features if result else []

def _get_manual_features(pipeline: Pipeline) -> List[str]:
    """Get manual features from previous step if available."""
    result = FeatureSelectionResult.objects.filter(pipeline=pipeline).first()
    return result.manual_features if result else []

def _scale_features(X: pd.DataFrame) -> pd.DataFrame:
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    return pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

def _generate_global_shap_plot(
    X: pd.DataFrame,
    shap_values: np.ndarray
) -> bytes:
    """Generate a global SHAP plot."""
    plt.figure(figsize=(14, 6), dpi=150)
    shap.summary_plot(
        shap_values,
        X,
        feature_names=X.columns,
        plot_type="bar",
        show=False
    )
    plt.xlabel(
        "mean(|SHAP value|) (impact on model output magnitude)",
        fontsize=12
    )
    plt.tick_params(axis='y', labelsize=10)
    plt.title("Важность признаков", fontsize=14)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

def _generate_local_shap_plot(
    X: pd.DataFrame,
    shap_values: np.ndarray
) -> bytes:
    """Generate a SHAP-distribution plot."""
    plt.figure(figsize=(14, 8), dpi=150)
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=X.columns, 
        show=False
    )
    plt.xlabel(
        "SHAP value (impact on model output)", 
        fontsize=12
    )
    plt.tick_params(axis='y', labelsize=10)
    plt.title(
        "Распределение SHAP-значений по признакам", 
        fontsize=14
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

# Filter Methods
def filter_variance_threshold(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    **kwargs
) -> FeatureSelectionResult:
    """Remove features with variance below a specified threshold."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_result(pipeline)

    all_features = pipeline.dataset.user_selected_features
    print(f"Filter selected features IN variance threshold: {len(all_features)} \n {all_features}")
    X_all = X[all_features]

    threshold = kwargs.get('threshold', DEFAULT_THRESHOLD)
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_all, y)
    selected_columns = X_all.columns[selector.get_support()]
    print(f"Filter selected features OUT variance threshold: {len(selected_columns)} \n {selected_columns.tolist()}", end="\n\n")
        
    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
            filtered_features=selected_columns.tolist()
        )

def filter_anova(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    **kwargs
) -> FeatureSelectionResult:
    """Select top k features based on ANOVA F-value."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_result(pipeline)

    all_features = pipeline.dataset.user_selected_features
    print(f"Filter selected features IN anova: {len(all_features)} \n {all_features}")
    X_all = X[all_features]
    X_scaled = _scale_features(X_all)
    
    k = kwargs.get('k', 'all')
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
        
    top_features = feature_scores[
        feature_scores['p_value'] < 0.05
        ]['feature'].tolist()
    print(f"Filter selected features OUT anova: {len(top_features)} \n {top_features}", end="\n\n")
        
    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
            filtered_features=top_features
        )

def filter_mutual_info(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    **kwargs
) -> FeatureSelectionResult:
    """Select top k features based on mutual information."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_result(pipeline)

    all_features = pipeline.dataset.user_selected_features
    print(f"Filter selected features IN mutual info: {len(all_features)} \n {all_features}")
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

    top_features = feature_scores[
        feature_scores['mutual_info'] > DEFAULT_THRESHOLD
    ]['feature'].tolist()

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=top_features
    )

def filter_mrmr(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    **kwargs
) -> FeatureSelectionResult:
    """Select features using minimum Redundancy Maximum Relevance (mRMR)."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_result(pipeline)

    all_features = pipeline.dataset.user_selected_features
    print(f"Filter selected features IN mRMR: {len(all_features)} \n {all_features}")
    X_all = X[all_features] 

    k = kwargs.get('k', 0.5 * X.shape[1])
    data = X_all.copy()
    data['target'] = y
    selected = mRMR(data, 'MIQ', k)
    selected = [col for col in selected if col != 'target']

    if not selected:
        raise FeatureSelectionError("No valid features selected by mRMR")
    print(f"Filter selected features OUT mRMR: {len(selected)} \n {selected}", end="\n\n")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=selected
    )

# Wrapper Methods
def wrapper_sfs_logreg(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    **kwargs
) -> FeatureSelectionResult:
    """Perform forward feature selection with Logistic Regression."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_result(pipeline)

    filtered_features = _get_filtered_features(pipeline)
    print(f"Wrapper selected features IN sfs logreg: {len(filtered_features)} \n {filtered_features}")
    X_filtered = X[filtered_features]
    X_scaled = _scale_features(X_filtered)
    
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='liblinear',
        max_iter=1000,
        random_state=DEFAULT_RANDOM_STATE,
    )
    
    k_features = kwargs.get('k_features', DEFAULT_K_FEATURES)
    sfs = SFS(
        model,
        k_features=k_features,
        forward=True,
        floating=False,
        scoring='roc_auc',
        verbose=2,
        cv=DEFAULT_CV
    )
    sfs.fit(X_scaled.values, y.values)
    selected_features = list(X_filtered.columns[list(sfs.k_feature_idx_)])
    print(f"Wrapper selected features OUT sfs logreg: {len(selected_features)} \n {selected_features}", end="\n\n")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=filtered_features,
        wrapped_features=selected_features
    )

def wrapper_sfs_tree(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    **kwargs
) -> FeatureSelectionResult:
    """Perform forward feature selection with Decision Tree."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_result(pipeline)

    filtered_features = _get_filtered_features(pipeline)
    print(f"Wrapper selected features IN sfs tree: {len(filtered_features)} \n {filtered_features}")
    X_filtered = X[filtered_features]

    model = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=None,
        random_state=DEFAULT_RANDOM_STATE
    )
    
    k_features = kwargs.get('k_features', DEFAULT_K_FEATURES)
    sfs = SFS(
        model,
        k_features=k_features,
        forward=True,
        floating=False,
        scoring='roc_auc',
        verbose=2,
        cv=DEFAULT_CV
    )
    sfs.fit(X_filtered.values, y.values)
    selected_features = list(X_filtered.columns[list(sfs.k_feature_idx_)])
    print(f"Wrapper selected features OUT sfs tree: {len(selected_features)} \n {selected_features}", end="\n\n")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=filtered_features,
        wrapped_features=selected_features
    )

def wrapper_rfe_logreg(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    **kwargs
) -> FeatureSelectionResult:
    """Perform recursive feature elimination with Logistic Regression."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_result(pipeline)

    filtered_features = _get_filtered_features(pipeline)
    print(f"Wrapper selected features IN rfe logreg: {len(filtered_features)} \n {filtered_features}")
    X_filtered = X[filtered_features]
    X_scaled = _scale_features(X_filtered)
    
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='liblinear',
        max_iter=1000,
        random_state=DEFAULT_RANDOM_STATE,
    )
    
    n_features = kwargs.get('n_features_to_select', None)
    rfe = RFE(estimator=model, n_features_to_select=n_features, verbose=2)
    rfe.fit(X_scaled, y)
    selected_features = list(X_filtered.columns[rfe.support_])
    print(f"Wrapper selected features OUT rfe logreg: {len(selected_features)} \n {selected_features}", end="\n\n")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=filtered_features,
        wrapped_features=selected_features
    )

def wrapper_rfe_tree(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    **kwargs
) -> FeatureSelectionResult:
    """Perform recursive feature elimination with Decision Tree."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_result(pipeline)

    filtered_features = _get_filtered_features(pipeline)
    print(f"Wrapper selected features IN rfe tree: {len(filtered_features)} \n {filtered_features}")
    X_filtered = X[filtered_features]

    model = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=None,
        random_state=DEFAULT_RANDOM_STATE
    )
    
    n_features = kwargs.get('n_features_to_select', None)
    rfe = RFE(estimator=model, n_features_to_select=n_features, verbose=2)
    rfe.fit(X_filtered, y)
    selected_features = list(X_filtered.columns[rfe.support_])
    print(f"Wrapper selected features OUT rfe tree: {len(selected_features)} \n {selected_features}", end="\n\n")

    return FeatureSelectionResult.objects.create(
        pipeline=pipeline,
        filtered_features=filtered_features,
        wrapped_features=selected_features
    )

# SHAP Methods
def shap_linear_explainer(
    model: Any,
    X: pd.DataFrame,
    **kwargs
) -> Tuple[shap.Explainer, np.ndarray]:
    """Compute SHAP values using LinearExplainer."""
    explainer = shap.LinearExplainer(model, X, **kwargs)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values

def shap_tree_explainer(
    model: Any,
    X: pd.DataFrame,
    **kwargs
) -> Tuple[shap.Explainer, np.ndarray]:
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model, **kwargs)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 2:
            return explainer, shap_values
        elif len(shap_values.shape) == 3:
            return explainer, shap_values[:, :, 1]
        else:
            raise ValueError(f"Unexpected shap_values shape: {shap_values.shape}")
    else:
        raise ValueError("Unexpected type for shap_values")

# Model Methods
def model_logistic_regression(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    **kwargs
) -> Tuple[PerformanceMetric, ShapExplanation]:
    """Train and evaluate a Logistic Regression model."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_metrics(pipeline)

    manual_features = _get_manual_features(pipeline)
    print(f"Model selected features IN logreg: {len(manual_features)} \n {manual_features}")
    X_manual = X[manual_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_manual,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE
    )
    
    X_train_scaled = _scale_features(X_train)
    X_test_scaled = _scale_features(X_test)
    
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='liblinear',
        max_iter=1000,
        random_state=DEFAULT_RANDOM_STATE,
        **kwargs
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    explainer, shap_values = shap_linear_explainer(model, X_test_scaled)

    global_plot = _generate_global_shap_plot(X_test_scaled, shap_values)
    local_plot = _generate_local_shap_plot(X_test_scaled, shap_values)

    metrics = {
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    print(f"Model metrics OUT logreg: \n {metrics}", end="\n\n")

    return (
        PerformanceMetric.objects.create(pipeline=pipeline, **metrics),
        ShapExplanation.objects.create(
            pipeline=pipeline,
            global_explanation_image=ContentFile(
                global_plot,
                name=f'global_shap_{pipeline.id}.png'
            ),
            local_explanation_image=ContentFile(
                local_plot,
                name=f'local_shap_{pipeline.id}.png'
            )
        )
    )

def model_xgb_linear(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    **kwargs
) -> Tuple[PerformanceMetric, ShapExplanation]:
    """Train and evaluate an XGBoost model with linear booster."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_metrics(pipeline)

    manual_features = _get_manual_features(pipeline)
    print(f"Model selected features IN xgb linear: {len(manual_features)} \n {manual_features}")
    X_manual = X[manual_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_manual,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE
    )
    
    X_train_scaled = _scale_features(X_train)
    X_test_scaled = _scale_features(X_test)
    
    model = XGBClassifier(
        booster='gblinear',
        eval_metric='auc',
        random_state=DEFAULT_RANDOM_STATE,
        **kwargs
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    explainer, shap_values = shap_linear_explainer(model, X_test_scaled)

    global_plot = _generate_global_shap_plot(X_test_scaled, shap_values)
    local_plot = _generate_local_shap_plot(X_test_scaled, shap_values)

    metrics = {
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    print(f"Model metrics OUT xgb linear: \n {metrics}", end="\n\n")

    return (
        PerformanceMetric.objects.create(pipeline=pipeline, **metrics),
        ShapExplanation.objects.create(
            pipeline=pipeline,
            global_explanation_image=ContentFile(
                global_plot,
                name=f'global_shap_{pipeline.id}.png'
            ),
            local_explanation_image=ContentFile(
                local_plot,
                name=f'local_shap_{pipeline.id}.png'
            )
        )
    )

def model_decision_tree(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    **kwargs
) -> Tuple[PerformanceMetric, ShapExplanation]:
    """Train and evaluate a Decision Tree model."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_metrics(pipeline)

    manual_features = _get_manual_features(pipeline)
    print(f"Model selected features IN dtree: {len(manual_features)} \n {manual_features}")
    X_manual = X[manual_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_manual,
        y,
        test_size=DEFAULT_TEST_SIZE,
            random_state=DEFAULT_RANDOM_STATE
        )
        
    model = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=None,
        random_state=DEFAULT_RANDOM_STATE,
        **kwargs
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    explainer, shap_values = shap_tree_explainer(model, X_test)

    global_plot = _generate_global_shap_plot(X_test, shap_values)
    local_plot = _generate_local_shap_plot(X_test, shap_values)

    metrics = {
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
    print(f"Model metrics OUT decision tree: \n {metrics}", end="\n\n")

    return (
        PerformanceMetric.objects.create(pipeline=pipeline, **metrics),
            ShapExplanation.objects.create(
            pipeline=pipeline,
                global_explanation_image=ContentFile(
                    global_plot, 
                name=f'global_shap_{pipeline.id}.png'
                ),
                local_explanation_image=ContentFile(
                    local_plot, 
                name=f'local_shap_{pipeline.id}.png'
            )
        )
    )

def model_xgb_tree(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
        **kwargs
) -> Tuple[PerformanceMetric, ShapExplanation]:
    """Train and evaluate an XGBoost model with tree booster."""
    if not pipeline.dataset or not pipeline.dataset.user_selected_features:
        return _create_empty_metrics(pipeline)

    manual_features = _get_manual_features(pipeline)
    print(f"Model selected features IN xgb tree: {len(manual_features)} \n {manual_features}")
    X_manual = X[manual_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_manual,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE
    )

    model = XGBClassifier(
        booster='gbtree',
        eval_metric='auc',
            random_state=DEFAULT_RANDOM_STATE,
            **kwargs
        )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    explainer, shap_values = shap_tree_explainer(model, X_test)

    global_plot = _generate_global_shap_plot(X_test, shap_values)
    local_plot = _generate_local_shap_plot(X_test, shap_values)

    metrics = {
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    print(f"Model metrics OUT xgb tree: \n {metrics}", end="\n\n")

    return (
        PerformanceMetric.objects.create(pipeline=pipeline, **metrics),
        ShapExplanation.objects.create(
            pipeline=pipeline,
            global_explanation_image=ContentFile(
                global_plot,
                name=f'global_shap_{pipeline.id}.png'
            ),
            local_explanation_image=ContentFile(
                local_plot,
                name=f'local_shap_{pipeline.id}.png'
            )
        )
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

def run_pipeline(pipeline: Pipeline, run_model: bool = True) -> None:
    """Run all pipeline analysis steps.
    
    This function runs feature selection, model evaluation, and SHAP analysis
    for a pipeline. It's a comprehensive function that can be called after
    pipeline configuration is complete.
    
    Args:
        pipeline: The pipeline to analyze
        run_model: Whether to run the model training step (default: True)
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
    print(f"Running pipeline with config: {config}", end="\n\n")
    
    # Validate configuration
    required_keys = ['model_method'] if run_model else []
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
            
    if config.get('filter_method') and config['filter_method'] not in FILTER_METHODS:
        raise ValueError(f"Invalid filter_method: {config['filter_method']}")
    if config.get('wrapper_method') and config['wrapper_method'] not in WRAPPER_METHODS:
        raise ValueError(f"Invalid wrapper_method: {config['wrapper_method']}")
    if run_model and config['model_method'] not in MODEL_METHODS:
        raise ValueError(f"Invalid model_method: {config['model_method']}")

    # Step 1: Filter Method (Optional)
    if not run_model and config.get('filter_method'):
        filter_func = FILTER_METHODS[config['filter_method']]
        filter_params = config.get('filter_params', {})
        filter_func(pipeline, X.copy(), y.copy(), **filter_params)

    # Step 2: Wrapper Method (Optional)
    if not run_model and config.get('wrapper_method'):
        wrapper_func = WRAPPER_METHODS[config['wrapper_method']]
        wrapper_params = config.get('wrapper_params', {})
        wrapper_func(pipeline, X.copy(), y.copy(), **wrapper_params)

    # Step 3: Model Training and SHAP Analysis (Optional)
    if run_model:
        model_func = MODEL_METHODS[config['model_method']]
        model_params = config.get('model_params', {})
        model_func(pipeline, X.copy(), y.copy(), **model_params)
