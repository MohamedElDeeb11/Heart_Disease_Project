# config.py
"""
إعدادات مشروع تشخيص أمراض القلب
"""

class Config:
    # إعدادات المسارات
    DATA_PATH = 'data/heart_disease.csv'
    MODELS_DIR = 'models/'
    RESULTS_DIR = 'results/'
    NOTEBOOKS_DIR = 'notebooks/'
    APP_DIR = 'app/'
    
    # إعدادات معالجة البيانات
    TARGET_COLUMN = 'target'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    PCA_VARIANCE_THRESHOLD = 0.95
    
    # إعدادات اختيار الميزات
    NUM_SELECTED_FEATURES = 8
    FEATURE_SELECTION_METHODS = ['random_forest', 'rfe', 'statistical']
    
    # إعدادات النماذج
    MODEL_NAMES = [
        'Logistic Regression',
        'Decision Tree', 
        'Random Forest',
        'SVM',
        'K-Nearest Neighbors',
        'Gradient Boosting'
    ]
    
    # إعدادات ضبط المعاملات
    HYPERPARAMETER_TUNING = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2'],
            'max_iter': [1000, 2000]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'degree': [2, 3, 4]
        }
    }
    
    # إعدادات التجميع
    CLUSTERING = {
        'MAX_CLUSTERS': 15,
        'CLUSTERING_METHODS': ['kmeans', 'hierarchical']
    }
    
    # إعدادات التطبيق
    APP_CONFIG = {
        'TITLE': 'نظام تشخيص أمراض القلب',
        'DESCRIPTION': 'نظام ذكي للتنبؤ بأمراض القلب باستخدام التعلم الآلي',
        'THEME': 'light'
    }

# إعدادات الألوان للرسوم البيانية
COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4', 
    'accent': '#45B7D1',
    'success': '#96CEB4',
    'warning': '#FFEAA7',
    'background': '#f8f9fa'
}

# معلومات المشروع
PROJECT_INFO = {
    'name': 'مشروع تشخيص أمراض القلب',
    'version': '1.0.0',
    'description': 'نظام متكامل للتنبؤ بأمراض القلب باستخدام تقنيات التعلم الآلي',
    'author': 'فريق الذكاء الاصطناعي',
    'date': '2024'
}