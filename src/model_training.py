# src/model_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        
    def prepare_data(self, df, target_col='target', test_size=0.2):
        """تحضير البيانات للتدريب"""
        print("📊 تحضير البيانات للتدريب...")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"📈 بيانات التدريب: {X_train.shape}")
        print(f"📊 بيانات الاختبار: {X_test.shape}")
        print(f"🎯 توزيع الفئات في التدريب: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"🎯 توزيع الفئات في الاختبار: {pd.Series(y_test).value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """تهيئة النماذج المختلفة"""
        print("🔄 تهيئة النماذج...")
        
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            'SVM': SVC(
                probability=True, random_state=self.random_state
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state
            )
        }
        
        return models
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """تدريب جميع النماذج"""
        print("🚀 بدء تدريب النماذج...")
        
        models = self.initialize_models()
        results = []
        
        for name, model in models.items():
            print(f"\n📚 تدريب نموذج: {name}")
            
            # تدريب النموذج
            model.fit(X_train, y_train)
            
            # التنبؤ
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # حساب المقاييس
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
            
            # التقاطع عبر k-fold
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # حفظ النتائج
            model_results = {
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUC-ROC': auc,
                'CV Mean': cv_mean,
                'CV Std': cv_std,
                'Model Object': model
            }
            
            results.append(model_results)
            
            print(f"   ✅ الدقة: {accuracy:.4f}")
            print(f"   ✅ F1-Score: {f1:.4f}")
            print(f"   ✅ AUC-ROC: {auc:.4f}")
            
            # حفظ النموذج إذا كان الأفضل
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = model
        
        # تحويل النتائج إلى DataFrame
        results_df = pd.DataFrame(results)
        self.results = results_df
        self.models = {row['Model']: row['Model Object'] for _, row in results_df.iterrows()}
        
        return results_df
    
    def plot_model_comparison(self):
        """مقارنة أداء النماذج"""
        if self.results.empty:
            print("⚠️ لا توجد نتائج للمقارنة")
            return
        
        print("\n📊 مقارنة أداء النماذج...")
        
        # ترتيب النتائج حسب الدقة
        results_sorted = self.results.sort_values('Accuracy', ascending=False)
        
        # رسم مقارنة الأداء
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # الدقة
        axes[0,0].barh(results_sorted['Model'], results_sorted['Accuracy'], color='skyblue')
        axes[0,0].set_title('مقارنة الدقة بين النماذج')
        axes[0,0].set_xlabel('الدقة')
        
        # F1-Score
        axes[0,1].barh(results_sorted['Model'], results_sorted['F1-Score'], color='lightcoral')
        axes[0,1].set_title('مقارنة F1-Score بين النماذج')
        axes[0,1].set_xlabel('F1-Score')
        
        # AUC-ROC
        axes[1,0].barh(results_sorted['Model'], results_sorted['AUC-ROC'], color='lightgreen')
        axes[1,0].set_title('مقارنة AUC-ROC بين النماذج')
        axes[1,0].set_xlabel('AUC-ROC')
        
        # التقاطع
        axes[1,1].barh(results_sorted['Model'], results_sorted['CV Mean'], 
                      xerr=results_sorted['CV Std'], color='gold')
        axes[1,1].set_title('متوسط التقاطع (5-fold)')
        axes[1,1].set_xlabel('متوسط الدقة')
        
        plt.tight_layout()
        plt.show()
        
        # عرض أفضل النماذج
        print("\n🎯 أفضل 3 نماذج:")
        top_3 = results_sorted.head(3)
        for _, model in top_3.iterrows():
            print(f"   {model['Model']}:")
            print(f"      📊 الدقة: {model['Accuracy']:.4f}")
            print(f"      🎯 F1-Score: {model['F1-Score']:.4f}")
            print(f"      📈 AUC-ROC: {model['AUC-ROC']:.4f}")
    
    def plot_confusion_matrices(self, X_test, y_test):
        """رسم مصفوفات الارتباك للنماذج"""
        print("\n📋 رسم مصفوفات الارتباك...")
        
        n_models = len(self.models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
        axes = axes.flatten()
        
        for i, (name, model) in enumerate(self.models.items()):
            if i < len(axes):
                # التنبؤ
                y_pred = model.predict(X_test)
                
                # مصفوفة الارتباك
                cm = confusion_matrix(y_test, y_pred)
                
                # الرسم
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                           xticklabels=['سليم', 'مريض'], 
                           yticklabels=['سليم', 'مريض'])
                axes[i].set_title(f'مصفوفة الارتباك - {name}')
                axes[i].set_xlabel('التنبؤ')
                axes[i].set_ylabel('الحقيقة')
        
        # إخفاء المحاور الفارغة
        for i in range(len(self.models), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, X_test, y_test):
        """رسم منحنيات ROC للنماذج"""
        print("\n📈 رسم منحنيات ROC...")
        
        plt.figure(figsize=(12, 8))
        
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        # خط المرجع
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='نموذج عشوائي')
        
        plt.xlabel('معدل الإيجابيات الكاذبة')
        plt.ylabel('معدل الإيجابيات الحقيقية')
        plt.title('منحنيات ROC للنماذج المختلفة')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save_models(self, models_path='models/'):
        """حفظ النماذج المدربة"""
        import os
        os.makedirs(models_path, exist_ok=True)
        
        print(f"💾 حفظ النماذج في {models_path}...")
        
        for name, model in self.models.items():
            # تنظيف اسم النموذج للملف
            filename = name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(models_path, filename)
            
            joblib.dump(model, filepath)
            print(f"   ✅ تم حفظ {name} في {filename}")
        
        # حفظ أفضل نموذج
        if self.best_model is not None:
            best_model_path = os.path.join(models_path, 'best_model.pkl')
            joblib.dump(self.best_model, best_model_path)
            print(f"   🏆 تم حفظ أفضل نموذج في best_model.pkl")
        
        # حفظ النتائج
        results_path = os.path.join(models_path, 'training_results.csv')
        self.results.to_csv(results_path, index=False)
        print(f"   📊 تم حفظ النتائج في training_results.csv")
    
    def train_complete_pipeline(self, df, target_col='target'):
        """خطة تدريب كاملة"""
        print("🎯 بدء خطة التدريب الكاملة...")
        
        # 1. تحضير البيانات
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        # 2. تدريب النماذج
        results_df = self.train_models(X_train, X_test, y_train, y_test)
        
        # 3. مقارنة النماذج
        self.plot_model_comparison()
        
        # 4. مصفوفات الارتباك
        self.plot_confusion_matrices(X_test, y_test)
        
        # 5. منحنيات ROC
        self.plot_roc_curves(X_test, y_test)
        
        # 6. حفظ النماذج
        self.save_models()
        
        print("\n🎉 تم الانتهاء من التدريب بنجاح!")
        return results_df

# دالة مساعدة
def train_all_models(df, target_col='target'):
    trainer = ModelTrainer()
    results = trainer.train_complete_pipeline(df, target_col)
    return results, trainer