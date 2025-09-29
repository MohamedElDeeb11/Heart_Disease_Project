# src/__init__.py
"""
حزمة مشروع تشخيص أمراض القلب
"""

__version__ = "1.0.0"
__author__ = "فريق الذكاء الاصطناعي"

from .data_preprocessing import DataPreprocessor, load_and_preprocess_data
from .eda_analysis import EDAAnalysis, perform_comprehensive_eda
from .feature_selection import FeatureSelector, perform_feature_selection
from .model_training import ModelTrainer, train_all_models
from .hyperparameter_tuning import HyperparameterTuner, perform_hyperparameter_tuning
from .unsupervised_learning import UnsupervisedLearning, perform_unsupervised_learning
from .utils import ProjectUtils, calculate_model_metrics

__all__ = [
    'DataPreprocessor',
    'EDAAnalysis', 
    'FeatureSelector',
    'ModelTrainer',
    'HyperparameterTuner',
    'UnsupervisedLearning',
    'ProjectUtils'
]