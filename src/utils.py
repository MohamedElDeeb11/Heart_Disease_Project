# src/utils.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

class ProjectUtils:
    def __init__(self):
        self.colors = {
            'primary': '#FF6B6B',
            'secondary': '#4ECDC4',
            'accent': '#45B7D1',
            'success': '#96CEB4',
            'warning': '#FFEAA7'
        }
    
    def create_project_structure(self):
        directories = [
            'data',
            'notebooks', 
            'src',
            'models',
            'app',
            'results/plots',
            'results/metrics'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 تم إنشاء مجلد: {directory}")
        
        print("✅ تم إنشاء هيكل المشروع بنجاح")
    
    def save_plot(self, fig, filename: str, dpi: int = 300):
        plots_dir = 'results/plots'
        os.makedirs(plots_dir, exist_ok=True)
        filepath = os.path.join(plots_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"💾 تم حفظ الرسم في: {filepath}")
    
    def save_metrics(self, metrics: dict, filename: str):
        metrics_dir = 'results/metrics'
        os.makedirs(metrics_dir, exist_ok=True)
        filepath = os.path.join(metrics_dir, filename)
        
        if filename.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4, ensure_ascii=False)
        elif filename.endswith('.csv'):
            if isinstance(metrics, dict):
                pd.DataFrame([metrics]).to_csv(filepath, index=False)
            else:
                metrics.to_csv(filepath, index=False)
        
        print(f"💾 تم حفظ المقاييس في: {filepath}")
    
    def load_model(self, model_path: str):
        try:
            model = joblib.load(model_path)
            print(f"✅ تم تحميل النموذج من: {model_path}")
            return model
        except Exception as e:
            print(f"❌ خطأ في تحميل النموذج: {e}")
            return None
    
    def save_model(self, model, model_path: str):
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
            print(f"✅ تم حفظ النموذج في: {model_path}")
        except Exception as e:
            print(f"❌ خطأ في حفظ النموذج: {e}")
    
    def plot_styling(self):
        plt.style.use('default')
        sns.set_palette(list(self.colors.values()))
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def print_project_info(self):
        info = """
        🏥 مشروع تشخيص أمراض القلب
        ===========================
        
        📊 المهام المنجزة:
        ✅ معالجة البيانات وتنظيفها
        ✅ التحليل الاستكشافي (EDA)
        ✅ اختيار الميزات المهمة
        ✅ تدريب نماذج التعلم الآلي
        ✅ ضبط معاملات النماذج
        ✅ التحليل غير المشرف
        ✅ تطبيق ويب تفاعلي
        
        🚀 كيفية التشغيل:
        1. python main.py (للتشغيل الكامل)
        2. cd app && streamlit run app.py (لتشغيل التطبيق)
        """
        print(info)

def calculate_model_metrics(y_true, y_pred, y_pred_proba=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

def get_feature_descriptions():
    descriptions = {
        'age': 'العمر بالسنوات',
        'sex': 'الجنس (1: ذكر, 0: أنثى)',
        'cp': 'نوع ألم الصدر (0-3)',
        'trestbps': 'ضغط الدم الانقباضي (mm Hg)',
        'chol': 'الكوليسترول (mg/dl)',
        'fbs': 'سكر الدم الصائم > 120 mg/dl (1: نعم, 0: لا)',
        'restecg': 'نتائج تخطيط القلب (0-2)',
        'thalach': 'أقصى معدل ضربات قلب',
        'exang': 'ذبحة صدرية مستحثة بالرياضة (1: نعم, 0: لا)',
        'oldpeak': 'اكتئاب ST الناتج عن الرياضة',
        'slope': 'ميل segment ST أثناء الذروة (1-3)',
        'ca': 'عدد الأوعية الرئيسية (0-3)',
        'thal': 'نوع الثلاسيميا (1-3)',
        'target': 'وجود مرض القلب (1: نعم, 0: لا)'
    }
    return descriptions

utils = ProjectUtils()