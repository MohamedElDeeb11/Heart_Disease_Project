# main.py
import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("🚀 مشروع تشخيص أمراض القلب - الإصدار الكامل")

def create_project_structure():
    """إنشاء هيكل المجلدات المطلوبة"""
    folders = ['data', 'models', 'results/plots', 'results/metrics']
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 تم إنشاء: {folder}")
    
    print("✅ تم إنشاء هيكل المشروع")

def check_requirements():
    """التحقق من المتطلبات"""
    try:
        import pandas as pd
        import sklearn
        import streamlit
        print("✅ جميع المتطلبات مثبتة")
        return True
    except ImportError as e:
        print(f"❌ متطلبات ناقصة: {e}")
        return False

def main():
    # إنشاء الهيكل
    create_project_structure()
    
    # التحقق من المتطلبات
    if not check_requirements():
        print("📥 يرجى تثبيت المتطلبات أولاً:")
        print("pip install -r requirements.txt")
        return
    
    # التحقق من وجود البيانات
    data_file = 'data/heart_disease.csv'
    if not os.path.exists(data_file):
        print(f"""
❌ ملف البيانات غير موجود: {data_file}

📥 يرجى تحميل البيانات من:
https://www.kaggle.com/datasets/ronitf/heart-disease-uci

وحفظها في: {data_file}
        """)
        return
    
    print("\n🎯 المشروع جاهز للتشغيل!")
    print("📊 لتشغيل خطوة معينة، افتح الملف المناسب في مجلد notebooks/")
    print("🌐 لتشغيل التطبيق: cd app && streamlit run app.py")
    print("🔧 لتدريب النماذج: python -c \"from src.model_training import train_all_models; train_all_models()\"")

if __name__ == "__main__":
    main()