import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from pymrmr import mRMR
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Filter Methods
def filter_variance_threshold(X, y=None, **kwargs) -> list:
    """Remove features with variance below a specified threshold."""
    threshold = kwargs.get('threshold', 0.1)
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X, y)
    selected_columns = X.columns[selector.get_support()]

    return selected_columns.tolist()


def filter_anova(X, y, **kwargs) -> list:
    """Select top k features based on ANOVA F-value."""
    k = kwargs.get('k', 'all')
    # Scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_scaled, y)

    scores = selector.scores_
    p_values = selector.pvalues_
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'f_score': scores,
        'p_value': p_values
    })
    print(feature_scores)

    top_features = feature_scores[feature_scores['p_value'] < 0.05]['feature'].tolist()

    return top_features


def filter_mutual_info(X, y, **kwargs) -> list:
    """Select top k features based on mutual information."""
    k = kwargs.get('k', 'all')
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    selector.fit(X, y)  
    scores = selector.scores_
    feature_scores = pd.DataFrame({
        'feature': X.columns,
        'mutual_info': scores,
    })
    print(feature_scores)

    top_features = feature_scores[feature_scores['mutual_info'] > 0.1]['feature'].tolist()

    return top_features


def filter_mrmr(X, y, **kwargs) -> list:
    """Select features using minimum Redundancy Maximum Relevance (mRMR)."""
    k = kwargs.get('k', 0.5 * X.shape[1])
    data = X.copy()
    data['target'] = y
    selected = mRMR(data, 'MIQ', k)
    # Remove 'target' from selected features if present
    selected = [col for col in selected if col != 'target']
    if not selected:
        raise ValueError("No valid features selected by mRMR")

    return selected


# Wrapper Methods
def wrapper_sfs_logreg(X, y, **kwargs) -> list:
    """Perform forward feature selection with Logistic Regression."""
    # Scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
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
    selected_features = list(X.columns[list(sfs.k_feature_idx_)])

    return selected_features


def wrapper_sfs_tree(X, y, **kwargs) -> list:
    """Perform forward feature selection with Decision Tree."""
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
    sfs.fit(X.values, y.values)
    selected_features = list(X.columns[list(sfs.k_feature_idx_)])

    return selected_features


def wrapper_rfe_logreg(X, y, **kwargs) -> list:
    """Perform recursive feature elimination with Logistic Regression."""
    n_features = kwargs.get('n_features_to_select', None)
    # Scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    model = LogisticRegression(
        penalty='l2',          # Use L2 regularization for stability
        C=1.0,                 # Moderate regularization strength
        solver='liblinear',    # Efficient solver for L2 penalty
        max_iter=1000,         # Ensure convergence for complex datasets
        random_state=42,
    )
    rfe = RFE(estimator=model, n_features_to_select=n_features, verbose=2)
    rfe.fit(X_scaled, y)
    selected_features = list(X.columns[rfe.support_])

    return selected_features


def wrapper_rfe_tree(X, y, **kwargs) -> list:
    """Perform recursive feature elimination with Decision Tree."""
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
    rfe.fit(X, y)
    selected_features = list(X.columns[rfe.support_])

    return selected_features


# SHAP Methods
def shap_linear_explainer(model, X, **kwargs) -> tuple:
    """Compute SHAP values using LinearExplainer."""
    explainer = shap.LinearExplainer(model, X, **kwargs)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def shap_tree_explainer(model, X, **kwargs) -> tuple:
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model, **kwargs)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


# Model Methods
def model_logistic_regression(X, y, **kwargs) -> tuple:
    """Train and evaluate a Logistic Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]),
        'f1': f1_score(y_test, y_pred),
        'shap_values': shap_values
    }
    return model, metrics


def model_xgb_linear(X, y, **kwargs) -> tuple:
    """Train and evaluate an XGBoost model with linear booster."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    model = XGBClassifier(booster='gblinear', eval_metric='auc', random_state=42, **kwargs)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    explainer, shap_values = shap_linear_explainer(model, X_test_scaled)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]),
        'f1': f1_score(y_test, y_pred),
        'shap_values': shap_values
    }
    return model, metrics


def model_decision_tree(X, y, **kwargs) -> tuple:
    """Train and evaluate a Decision Tree model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        'f1': f1_score(y_test, y_pred),
        'shap_values': shap_values
    }
    return model, metrics


def model_xgb_tree(X, y, **kwargs) -> tuple:
    """Train and evaluate an XGBoost model with tree booster."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(booster='gbtree', eval_metric='auc', random_state=42, **kwargs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    explainer, shap_values = shap_tree_explainer(model, X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        'f1': f1_score(y_test, y_pred),
        'shap_values': shap_values
    }
    return model, metrics


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


# Pipeline Function
def run_pipeline(X, y, config) -> dict:
    """
    Runs a feature selection and modeling pipeline based on a configuration dictionary.

    Args:
        X (pd.DataFrame): Input features
        y (pd.Series): Target variable
        config (dict): Configuration specifying methods and parameters for each step

    Returns:
        dict: Results including selected features, model, metrics, and SHAP values

    Raises:
        ValueError: If configuration contains invalid methods or parameters
    """
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

    results = {}
    current_features = X.columns.tolist()
    print(f"Current features: {current_features}")

    # Step 1: Filter Method (Optional)
    if 'filter_method' in config and config['filter_method']:
        filter_func = FILTER_METHODS[config['filter_method']]
        filter_params = config.get('filter_params', {})
        filtered_features = filter_func(X.copy()[current_features], y, **filter_params)
        current_features = filtered_features
        results['filtered_features'] = filtered_features
    else:
        results['filtered_features'] = current_features
    print(f"Filtered features: {current_features}")

    # Step 2: Wrapper Method (Optional)
    if 'wrapper_method' in config and config['wrapper_method']:
        wrapper_func = WRAPPER_METHODS[config['wrapper_method']]
        wrapper_params = config.get('wrapper_params', {})
        selected_features = wrapper_func(X.copy()[current_features], y, **wrapper_params)
        current_features = selected_features
        results['selected_features'] = selected_features
    else:
        results['selected_features'] = current_features
    print(f"Selected features: {current_features}")

    # Step 3: Model Training and SHAP Analysis
    model_func = MODEL_METHODS[config['model_method']]
    model_params = config.get('model_params', {})
    model, metrics = model_func(X.copy()[current_features], y, **model_params)
    results['model'] = model
    results['metrics'] = metrics

    return results


# Example Usage
if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_csv(r'./toy_data.csv')
        df.drop(columns=['Unnamed: 0'], inplace=True)
        X = df.drop('fibroids', axis=1)
        y = df['fibroids']
        print(f"Initial dataset has {X.shape[1]} features")
    except FileNotFoundError:
        print("Error: Data file not found.")
        exit(1)
    except KeyError as e:
        print(f"Error: {e}")
        exit(1)

    # Define configuration
    config = {
        'filter_method': 'anova',               # 'variance_threshold', 'anova', 'mutual_info', 'mrmr'
        'wrapper_method': 'rfe_tree',           # 'rfe_tree', 'sfs_tree', 'sfs_logreg', 'rfe_logreg'
        'model_method': 'dtree',                # 'logreg', 'xgb_linear', 'dtree', 'xgb_tree'

        # 'filter_params': {'threshold': 0.1},        # {'k': 10}
        # 'wrapper_params': {'k_features': 'best'},   # {'n_features_to_select': 0.5}
        # 'model_params': {'C': 1.0, 'penalty': 'l1'}    # {'C': 1.0, 'penalty': 'l1'}, {'max_depth': 3}
    }

    # Run pipeline
    try:
        results = run_pipeline(X, y, config)
        print("Filtered Features:", results['filtered_features'])
        print("Selected Features:", results['selected_features'])
        print("Metrics:", results['metrics'])
    except Exception as e:
        print(f"Pipeline error: {e}")
