"""Forms for the Featurama application.

This module defines Django forms for validating user input.
"""

from django import forms
from .models import Pipeline
from .services import MethodsService


class DatasetUploadForm(forms.Form):
    """Form for validating dataset file uploads."""
    
    ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'xls']
    
    dataset_file = forms.FileField(
        required=True,
        widget=forms.ClearableFileInput(
            attrs={'accept': '.csv,.xlsx,.xls'}
        )
    )
    
    def clean_dataset_file(self):
        """Validate the uploaded file."""
        file = self.cleaned_data['dataset_file']
        
        # Check file extension
        extension = file.name.split('.')[-1].lower()
        if extension not in self.ALLOWED_EXTENSIONS:
            raise forms.ValidationError(
                f"Неподдерживаемый формат файла. Пожалуйста, загрузите один из: "
                f"{', '.join(self.ALLOWED_EXTENSIONS)}."
            )
            
        # Check file size (10MB max)
        if file.size > 10 * 1024 * 1024:
            raise forms.ValidationError(
                "Файл слишком большой. Максимальный размер 10MB."
            )
            
        return file


class TargetVariableForm(forms.Form):
    """Form for selecting a target variable."""
    
    target_variable = forms.CharField(
        required=True,
        widget=forms.Select()
    )
    
    def __init__(self, *args, features=None, **kwargs):
        """Initialize the form with dynamic choices."""
        super().__init__(*args, **kwargs)
        
        if features:
            self.fields['target_variable'].widget.choices = [
                ('', 'Выберите целевую переменную')
            ] + [(f, f) for f in features]


class FeatureSelectionForm(forms.Form):
    """Form for selecting features from a dataset."""
    
    selected_features = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple(
            attrs={'class': 'feature-checkbox'}
        )
    )
    
    def __init__(self, *args, features=None, target_variable=None, **kwargs):
        """Initialize the form with available features.
        
        Args:
            features: List of all available features in the dataset
            target_variable: Target variable to exclude from selection options
        """
        super().__init__(*args, **kwargs)
        
        if features and target_variable:
            # Filter out the target variable from selectable features
            available_features = [f for f in features if f != target_variable]
            choices = [(f, f) for f in available_features]
            self.fields['selected_features'].choices = choices
            
            # Store original features for reference
            self.all_features = features
            self.target_variable = target_variable


class FilterMethodConfigForm(forms.Form):
    """Form for configuring filter method hyperparameters."""
    
    # Variance Threshold parameters
    threshold = forms.FloatField(
        required=False,
        min_value=0.0,
        widget=forms.NumberInput(attrs={'step': '0.01', 'class': 'form-control'}),
        help_text="Варьирование значений ниже порога будет удалено, по умолчанию: 0.1"
    )
    
    # ANOVA parameters
    k_anova = forms.IntegerField(
        required=False,
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text="Количество лучших признаков для отбора"
    )
    
    # Mutual Information parameters
    k_mutual_info = forms.IntegerField(
        required=False,
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text="Количество лучших признаков для отбора"
    )
    
    # mRMR parameters
    n_features = forms.IntegerField(
        required=False,
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text="Количество признаков для отбора"
    )
    
    method = forms.ChoiceField(
        required=False,
        choices=[('MID', 'MID - Mutual Information Difference'),
                 ('MIQ', 'MIQ - Mutual Information Quotient')],
        widget=forms.RadioSelect(attrs={'class': 'form-check-input'}),
        help_text="Метод расчета"
    )
    
    def clean(self):
        """Validate that parameters correspond to selected method."""
        cleaned_data = super().clean()
        filter_method = self.data.get('filter_method')
        
        # Only validate parameters for the selected method
        if filter_method == 'Variance Threshold':
            if 'threshold' not in cleaned_data or cleaned_data['threshold'] is None:
                cleaned_data['threshold'] = 0.1
                
        elif filter_method == 'ANOVA':
            if 'k_anova' not in cleaned_data or not cleaned_data['k_anova']:
                # Will use default in the algorithm
                pass
                
        elif filter_method == 'Mutual Information':
            if 'k_mutual_info' not in cleaned_data or not cleaned_data['k_mutual_info']:
                # Will use default in the algorithm
                pass
                
        elif filter_method == 'MRMR':
            if 'n_features' not in cleaned_data or not cleaned_data['n_features']:
                # Will use default in the algorithm
                pass
            
            if 'method' not in cleaned_data or not cleaned_data['method']:
                cleaned_data['method'] = 'MID'
                
        return cleaned_data
    
    def get_filter_params(self, filter_method):
        """Get the parameters for the selected filter method as a dict."""
        cleaned_data = self.cleaned_data
        params = {}
        
        if filter_method == 'Variance Threshold':
            params['threshold'] = cleaned_data.get('threshold', 0.1)
            
        elif filter_method == 'ANOVA':
            if cleaned_data.get('k_anova'):
                params['k'] = cleaned_data['k_anova']
                
        elif filter_method == 'Mutual Information':
            if cleaned_data.get('k_mutual_info'):
                params['k'] = cleaned_data['k_mutual_info']
                
        elif filter_method == 'MRMR':
            if cleaned_data.get('n_features'):
                params['n_features'] = cleaned_data['n_features']
                
            if cleaned_data.get('method'):
                params['method'] = cleaned_data['method']
                
        return params


class WrapperMethodConfigForm(forms.Form):
    """Form for configuring wrapper method hyperparameters."""
    
    # SFS with Logistic Regression parameters
    scoring_logreg = forms.ChoiceField(
        required=False,
        choices=[
            ('accuracy', 'Accuracy - Общая точность'),
            ('f1', 'F1 - Среднее гармоническое точности и полноты'),
            ('precision', 'Precision - Точность (доля истинно положительных)'),
            ('recall', 'Recall - Полнота (чувствительность)'),
            ('roc_auc', 'ROC AUC - Площадь под ROC-кривой')
        ],
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text="Метрика для оценки качества модели"
    )
    
    # SFS with Decision Tree parameters
    scoring_tree = forms.ChoiceField(
        required=False,
        choices=[
            ('accuracy', 'Accuracy - Общая точность'),
            ('f1', 'F1 - Среднее гармоническое точности и полноты'),
            ('precision', 'Precision - Точность (доля истинно положительных)'),
            ('recall', 'Recall - Полнота (чувствительность)'),
            ('roc_auc', 'ROC AUC - Площадь под ROC-кривой')
        ],
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text="Метрика для оценки качества модели"
    )
    
    # RFE with Logistic Regression parameters
    n_features_logreg = forms.FloatField(
        required=False,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': '0.01', 'class': 'form-control'}),
        help_text="Доля признаков для отбора (0.0-1.0) или количество признаков, по умолчанию: 0.5"
    )
    
    # RFE with Decision Tree parameters
    n_features_tree = forms.FloatField(
        required=False,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': '0.01', 'class': 'form-control'}),
        help_text="Доля признаков для отбора (0.0-1.0) или количество признаков, по умолчанию: 0.5"
    )
    
    def clean(self):
        """Validate that parameters correspond to selected method."""
        cleaned_data = super().clean()
        wrapper_method = self.data.get('wrapper_method')
        
        # Only validate parameters for the selected method
        if wrapper_method == 'SFS with Logistic Regression':
            if 'scoring_logreg' not in cleaned_data or not cleaned_data['scoring_logreg']:
                cleaned_data['scoring_logreg'] = 'accuracy'
                
        elif wrapper_method == 'SFS with Decision Tree':
            if 'scoring_tree' not in cleaned_data or not cleaned_data['scoring_tree']:
                cleaned_data['scoring_tree'] = 'accuracy'
                
        elif wrapper_method == 'RFE with Logistic Regression':
            if 'n_features_logreg' not in cleaned_data or cleaned_data['n_features_logreg'] is None:
                cleaned_data['n_features_logreg'] = 0.5
                
        elif wrapper_method == 'RFE with Decision Tree':
            if 'n_features_tree' not in cleaned_data or cleaned_data['n_features_tree'] is None:
                cleaned_data['n_features_tree'] = 0.5
                
        return cleaned_data
    
    def get_wrapper_params(self, wrapper_method):
        """Get the parameters for the selected wrapper method as a dict."""
        cleaned_data = self.cleaned_data
        params = {}
        
        if wrapper_method == 'SFS with Logistic Regression':
            if cleaned_data.get('scoring_logreg'):
                params['scoring'] = cleaned_data['scoring_logreg']
                
        elif wrapper_method == 'SFS with Decision Tree':
            if cleaned_data.get('scoring_tree'):
                params['scoring'] = cleaned_data['scoring_tree']
                
        elif wrapper_method == 'RFE with Logistic Regression':
            if cleaned_data.get('n_features_logreg') is not None:
                params['n_features_to_select'] = cleaned_data['n_features_logreg']
                
        elif wrapper_method == 'RFE with Decision Tree':
            if cleaned_data.get('n_features_tree') is not None:
                params['n_features_to_select'] = cleaned_data['n_features_tree']
                
        return params


class ModelMethodConfigForm(forms.Form):
    """Form for configuring model hyperparameters."""
    
    # Common parameters
    test_size = forms.FloatField(
        required=False,
        min_value=0.01,
        max_value=0.99,
        widget=forms.NumberInput(attrs={'step': '0.05', 'class': 'form-control'}),
        help_text="Доля данных для тестирования (от 0.01 до 0.99), по умолчанию: 0.25"
    )
    
    # Logistic Regression parameters
    C = forms.FloatField(
        required=False,
        min_value=0.0001,
        widget=forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
        help_text="Параметр регуляризации (меньше = сильнее регуляризация), по умолчанию: 1.0"
    )
    
    penalty = forms.ChoiceField(
        required=False,
        choices=[
            ('l1', 'L1 - LASSO (абсолютные значения)'),
            ('l2', 'L2 - Ridge (квадраты)'),
            ('elasticnet', 'ElasticNet (комбинация L1 и L2)'),
            ('none', 'Без регуляризации')
        ],
        widget=forms.Select(attrs={'class': 'form-control', 'id': 'id_penalty'}),
        help_text="Тип регуляризации, по умолчанию: L2"
    )
    
    solver = forms.ChoiceField(
        required=False,
        choices=[
            ('lbfgs', 'LBFGS (по умолчанию)'),
            ('liblinear', 'LibLinear (для малых датасетов)'),
            ('newton-cg', 'Newton-CG (для больших датасетов)'),
            ('sag', 'SAG (для больших датасетов)'),
            ('saga', 'SAGA (для больших датасетов, все типы регуляризации)')
        ],
        widget=forms.Select(attrs={'class': 'form-control', 'id': 'id_solver'}),
        help_text="Алгоритм оптимизации, некоторые солверы доступны только для определенных типов регуляризации"
    )
    
    # XGBoost parameters
    n_estimators = forms.IntegerField(
        required=False,
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text="Количество деревьев, по умолчанию: 100"
    )
    
    learning_rate = forms.FloatField(
        required=False,
        min_value=0.001,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': '0.01', 'class': 'form-control'}),
        help_text="Скорость обучения (от 0.001 до 1.0), по умолчанию: 0.3"
    )
    
    # Decision Tree parameters
    max_depth = forms.IntegerField(
        required=False,
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text="Максимальная глубина дерева (оставьте пустым для неограниченной глубины)"
    )
    
    min_samples_split = forms.FloatField(
        required=False,
        min_value=0.0001,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': '0.01', 'class': 'form-control'}),
        help_text="Минимальная доля образцов для разделения узла, по умолчанию: 0.01"
    )
    
    min_samples_leaf = forms.FloatField(
        required=False,
        min_value=0.0001,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'step': '0.01', 'class': 'form-control'}),
        help_text="Минимальная доля образцов в листе, по умолчанию: 0.01"
    )
    
    criterion = forms.ChoiceField(
        required=False,
        choices=[
            ('gini', 'Gini - индекс Джини (по умолчанию)'),
            ('entropy', 'Entropy - энтропия'),
            ('log_loss', 'Log Loss - функция логарифмических потерь')
        ],
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text="Функция для измерения качества разделения"
    )
    
    def clean(self):
        """Validate that parameters correspond to selected method."""
        cleaned_data = super().clean()
        model_method = self.data.get('model_method')
        
        # Validate common parameters
        if 'test_size' not in cleaned_data or cleaned_data['test_size'] is None:
            cleaned_data['test_size'] = 0.25
        
        # Validate based on model method
        if model_method == 'Logistic Regression':
            # Validate C parameter
            if 'C' not in cleaned_data or cleaned_data['C'] is None:
                cleaned_data['C'] = 1.0
                
            # Validate penalty parameter
            if 'penalty' not in cleaned_data or not cleaned_data['penalty']:
                cleaned_data['penalty'] = 'l2'
                
            # Validate solver parameter
            if 'solver' not in cleaned_data or not cleaned_data['solver']:
                cleaned_data['solver'] = 'lbfgs'
                
            # Validate solver compatibility with penalty
            if 'penalty' in cleaned_data and 'solver' in cleaned_data:
                valid_solvers = self._get_valid_solvers(cleaned_data['penalty'])
                if cleaned_data['solver'] not in valid_solvers:
                    cleaned_data['solver'] = valid_solvers[0]
        
        elif model_method in ['XGBoost Linear', 'XGBoost Tree']:
            # Validate n_estimators parameter
            if 'n_estimators' not in cleaned_data or cleaned_data['n_estimators'] is None:
                cleaned_data['n_estimators'] = 100
                
            # Validate learning_rate parameter
            if 'learning_rate' not in cleaned_data or cleaned_data['learning_rate'] is None:
                cleaned_data['learning_rate'] = 0.3
                
        elif model_method == 'Decision Tree':
            # max_depth can be null, no need to validate
            
            # Validate min_samples_split parameter
            if 'min_samples_split' not in cleaned_data or cleaned_data['min_samples_split'] is None:
                cleaned_data['min_samples_split'] = 0.01
                
            # Validate min_samples_leaf parameter
            if 'min_samples_leaf' not in cleaned_data or cleaned_data['min_samples_leaf'] is None:
                cleaned_data['min_samples_leaf'] = 0.01
                
            # Validate criterion parameter
            if 'criterion' not in cleaned_data or not cleaned_data['criterion']:
                cleaned_data['criterion'] = 'gini'
                
        return cleaned_data
    
    def _get_valid_solvers(self, penalty):
        """Get valid solvers for a given penalty."""
        penalty_solvers = {
            'l1': ['liblinear', 'saga'],
            'l2': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'elasticnet': ['saga'],
            'none': ['newton-cg', 'lbfgs', 'sag', 'saga']
        }
        return penalty_solvers.get(penalty, ['lbfgs'])
    
    def get_model_params(self, model_method):
        """Get the parameters for the selected model method as a dict."""
        cleaned_data = self.cleaned_data
        params = {}
        
        # Add common parameters
        if cleaned_data.get('test_size') is not None:
            params['test_size'] = cleaned_data['test_size']
        
        # Add model-specific parameters
        if model_method == 'Logistic Regression':
            if cleaned_data.get('C') is not None:
                params['C'] = cleaned_data['C']
                
            if cleaned_data.get('penalty'):
                params['penalty'] = cleaned_data['penalty']
                
            if cleaned_data.get('solver'):
                params['solver'] = cleaned_data['solver']
                
        elif model_method in ['XGBoost Linear', 'XGBoost Tree']:
            if cleaned_data.get('n_estimators') is not None:
                params['n_estimators'] = cleaned_data['n_estimators']
                
            if cleaned_data.get('learning_rate') is not None:
                params['learning_rate'] = cleaned_data['learning_rate']
                
        elif model_method == 'Decision Tree':
            if cleaned_data.get('max_depth') is not None:
                params['max_depth'] = cleaned_data['max_depth']
                
            if cleaned_data.get('min_samples_split') is not None:
                params['min_samples_split'] = cleaned_data['min_samples_split']
                
            if cleaned_data.get('min_samples_leaf') is not None:
                params['min_samples_leaf'] = cleaned_data['min_samples_leaf']
                
            if cleaned_data.get('criterion'):
                params['criterion'] = cleaned_data['criterion']
                
        return params


class PipelineConfigForm(forms.ModelForm):
    """Form for configuring pipeline methods."""
    
    class Meta:
        model = Pipeline
        fields = ['filter_method', 'wrapper_method', 'model_method']
        
    def __init__(self, *args, **kwargs):
        """Initialize the form with dynamic choices."""
        super().__init__(*args, **kwargs)
        
        # Get available methods
        methods = MethodsService.get_available_methods()
        
        # Set choices for each method field
        self.fields['filter_method'].widget = forms.Select(choices=[
            ('', 'Выберите метод фильтрации')
        ] + [(k, k.replace('_', ' ').title()) 
             for k in methods['filter_methods']])
        
        self.fields['wrapper_method'].widget = forms.Select(choices=[
            ('', 'Выберите метод обертки')
        ] + [(k, k.replace('_', ' ').title()) 
             for k in methods['wrapper_methods']])
        
        self.fields['model_method'].widget = forms.Select(choices=[
            ('', 'Выберите метод модели')
        ] + [(k, k.replace('_', ' ').title()) 
             for k in methods['model_methods']])
        
        # Mark all fields as required
        for field in self.fields.values():
            field.required = True 