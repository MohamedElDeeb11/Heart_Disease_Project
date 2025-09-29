# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os

# إضافة مسار src للمكتبات
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor
from eda_analysis import EDAAnalysis
from model_training import ModelTrainer

class HeartDiseaseApp:
    def __init__(self):
        self.set_page_config()
        
    def set_page_config(self):
        """إعدادات صفحة Streamlit"""
        st.set_page_config(
            page_title="نظام تشخيص أمراض القلب",
            page_icon="❤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # تخصيص التصميم
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #FF6B6B;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #4ECDC4;
            margin-bottom: 1rem;
        }
        .prediction-card {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_models(self):
        """تحميل النماذج المدربة"""
        try:
            models_path = os.path.join('models', 'best_model.pkl')
            self.model = joblib.load(models_path)
            st.success("✅ تم تحميل النموذج بنجاح")
            return True
        except Exception as e:
            st.error(f"❌ خطأ في تحميل النموذج: {e}")
            return False
    
    def show_header(self):
        """عرض رأس التطبيق"""
        st.markdown('<h1 class="main-header">❤️ نظام تشخيص أمراض القلب باستخدام الذكاء الاصطناعي</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("""
            **ملاحظة:** هذا النظام يستخدم خوارزميات التعلم الآلي للتنبؤ باحتمالية 
            الإصابة بأمراض القلب بناءً على المعطيات الصحية للمريض.
            """)
    
    def input_features(self):
        """إدخال معطيات المريض"""
        st.markdown('<h2 class="sub-header">📋 إدخال البيانات الصحية</h2>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("العمر", 20, 100, 50)
            trestbps = st.slider("ضغط الدم الانقباضي", 90, 200, 120)
            chol = st.slider("الكوليسترول", 100, 600, 200)
            
        with col2:
            thalach = st.slider("أقصى معدل ضربات قلب", 60, 220, 150)
            oldpeak = st.slider("اكتئاب ST", 0.0, 6.0, 1.0)
            ca = st.slider("عدد الأوعية الرئيسية", 0, 4, 1)
            
        with col3:
            sex = st.selectbox("الجنس", ["ذكر", "أنثى"])
            cp = st.selectbox("نوع ألم الصدر", [
                "نوع 0: لا يوجد", 
                "نوع 1: ذبحة صدرية نموذجية",
                "نوع 2: ذبحة صدرية غير نموذجية",
                "نوع 3: ألم غير مرتبط بالقلب"
            ])
            fbs = st.selectbox("سكر الدم الصائم", ["≤ 120 mg/dl", "> 120 mg/dl"])
            exang = st.selectbox("ذبحة صدرية مستحثة بالرياضة", ["لا", "نعم"])
        
        # تحويل المدخلات إلى صيغة رقمية
        input_data = {
            'age': age,
            'sex': 1 if sex == "ذكر" else 0,
            'cp': cp.split(":")[0][-1],  # استخراج رقم النوع
            'trestbps': trestbps,
            'chol': chol,
            'fbs': 1 if fbs == "> 120 mg/dl" else 0,
            'restecg': 0,  # قيمة افتراضية
            'thalach': thalach,
            'exang': 1 if exang == "نعم" else 0,
            'oldpeak': oldpeak,
            'slope': 1,  # قيمة افتراضية
            'ca': ca,
            'thal': 2  # قيمة افتراضية
        }
        
        return input_data
    
    def make_prediction(self, input_data):
        """إجراء التنبؤ"""
        try:
            # تحويل البيانات إلى DataFrame
            input_df = pd.DataFrame([input_data])
            
            # التنبؤ
            prediction = self.model.predict(input_df)[0]
            probability = self.model.predict_proba(input_df)[0]
            
            return prediction, probability
        except Exception as e:
            st.error(f"❌ خطأ في التنبؤ: {e}")
            return None, None
    
    def show_prediction_result(self, prediction, probability):
        """عرض نتيجة التنبؤ"""
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 نتيجة التشخيص</h2>', 
                   unsafe_allow_html=True)
        
        if prediction == 1:
            st.error(f"""
            ## 🚨 نتيجة إيجابية
            **التشخيص:** يوجد مؤشرات على مرض القلب
            **درجة الثقة:** {probability[1]:.2%}
            **التوصية:** ننصح بمراجعة طبيب القلب لأجراء فحوصات أكثر دقة
            """)
        else:
            st.success(f"""
            ## ✅ نتيجة سلبية
            **التشخيص:** لا توجد مؤشرات على مرض القلب
            **درجة الثقة:** {probability[0]:.2%}
            **التوصية:** حافظ على نمط حياة صحي ومارس الرياضة بانتظام
            """)
        
        # عرض احتمالات التفصيل
        col1, col2 = st.columns(2)
        with col1:
            st.metric("احتمالية السلامة", f"{probability[0]:.2%}")
        with col2:
            st.metric("احتمالية المرض", f"{probability[1]:.2%}")
    
    def show_data_analysis(self):
        """عرض تحليل البيانات"""
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📈 تحليل البيانات</h2>', 
                   unsafe_allow_html=True)
        
        try:
            # تحميل البيانات
            data_path = os.path.join('data', 'heart_disease.csv')
            df = pd.read_csv(data_path)
            
            tab1, tab2, tab3 = st.tabs(["نظرة عامة", "التوزيعات", "الارتباطات"])
            
            with tab1:
                st.subheader("نظرة عامة على البيانات")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("عدد المرضى", len(df))
                with col2:
                    st.metric("عدد المصابين", df['target'].sum())
                with col3:
                    st.metric("نسبة الإصابة", f"{(df['target'].sum()/len(df))*100:.1f}%")
                
                st.dataframe(df.head(10))
            
            with tab2:
                st.subheader("توزيع البيانات")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # توزيع العمر
                    fig, ax = plt.subplots()
                    df['age'].hist(bins=20, ax=ax, color='#4ECDC4')
                    ax.set_title('توزيع العمر')
                    ax.set_xlabel('العمر')
                    ax.set_ylabel('التكرار')
                    st.pyplot(fig)
                
                with col2:
                    # توزيع المرض
                    fig, ax = plt.subplots()
                    df['target'].value_counts().plot(kind='bar', ax=ax, color=['#FF6B6B', '#45B7D1'])
                    ax.set_title('توزيع حالات المرض')
                    ax.set_xlabel('الحالة (0: سليم, 1: مريض)')
                    ax.set_ylabel('عدد الحالات')
                    st.pyplot(fig)
            
            with tab3:
                st.subheader("مصفوفة الارتباط")
                
                # حساب مصفوفة الارتباط
                corr_matrix = df.corr()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                ax.set_title('مصفوفة الارتباط بين المتغيرات')
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"❌ خطأ في تحميل البيانات: {e}")
    
    def show_model_info(self):
        """عرض معلومات عن النموذج"""
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🤖 معلومات النموذج</h2>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **الخوارزمية المستخدمة:** Random Forest
            **الدقة:** ~85%
            **الميزات المستخدمة:** 13 ميزة طبية
            **الغرض:** تشخيص أمراض القلب
            """)
        
        with col2:
            st.info("""
            **ملاحظات مهمة:**
            - هذا النظام لأغراض بحثية وتعليمية
            - لا يغني عن استشارة الطبيب المختص
            - الدقة قد تختلف حسب جودة البيانات المدخلة
            """)
    
    def run(self):
        """تشغيل التطبيق"""
        self.show_header()
        
        # تحميل النماذج
        if not self.load_models():
            return
        
        # إنشاء تبويبات
        tab1, tab2, tab3 = st.tabs(["التشخيص", "تحليل البيانات", "معلومات"])
        
        with tab1:
            # إدخال البيانات
            input_data = self.input_features()
            
            # زر التنبؤ
            if st.button("🔍 إجراء التشخيص", type="primary", use_container_width=True):
                with st.spinner("جاري تحليل البيانات..."):
                    prediction, probability = self.make_prediction(input_data)
                    if prediction is not None:
                        self.show_prediction_result(prediction, probability)
        
        with tab2:
            self.show_data_analysis()
        
        with tab3:
            self.show_model_info()

# تشغيل التطبيق
if __name__ == "__main__":
    app = HeartDiseaseApp()
    app.run()