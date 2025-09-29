# src/hyperparameter_tuning.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class HyperparameterTuner:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_models = {}
        self.tuning_results = {}
        
    def define_search_spaces(self):
        """تعريف مساحات البحث للمعاملات"""
        print("🎯 تعريف مساحات البحث للمعاملات...")
        
        param_spaces = {
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
        
        return param_spaces
    
    def perform_grid_search(self, model, param_grid, X_train, y_train, model_name):
        """تنفيذ Grid Search"""
        print(f"🔍 تنفيذ Grid Search لـ {model_name}...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"   ✅ أفضل معاملات: {grid_search.best_params_}")
        print(f"   ✅ أفضل دقة: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def perform_randomized_search(self, model, param_dist, X_train, y_train, model_name, n_iter=50):
        """تنفيذ Randomized Search"""
        print(f"🎲 تنفيذ Randomized Search لـ {model_name}...")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        print(f"   ✅ أفضل معاملات: {random_search.best_params_}")
        print(f"   ✅ أفضل دقة: {random_search.best_score_:.4f}")
        
        return random_search
    
    def tune_models(self, X_train, X_test, y_train, y_test, use_randomized=True):
        """ضبط معاملات النماذج"""
        print("🚀 بدء ضبط معاملات النماذج...")
        
        param_spaces = self.define_search_spaces()
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state),
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'SVM': SVC(probability=True, random_state=self.random_state)
        }
        
        tuning_results = {}
        
        for model_name, model in models.items():
            print(f"\n🎯 معالجة نموذج: {model_name}")
            
            if use_randomized:
                # استخدام Randomized Search للسرعة
                search_result = self.perform_randomized_search(
                    model, param_spaces[model_name], X_train, y_train, model_name
                )
            else:
                # استخدام Grid Search للدقة
                search_result = self.perform_grid_search(
                    model, param_spaces[model_name], X_train, y_train, model_name
                )
            
            # أفضل نموذج
            best_model = search_result.best_estimator_
            
            # التقيم على بيانات الاختبار
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            test_accuracy = search_result.score(X_test, y_test)
            
            # حفظ النتائج
            tuning_results[model_name] = {
                'best_model': best_model,
                'best_params': search_result.best_params_,
                'best_score': search_result.best_score_,
                'test_accuracy': test_accuracy,
                'search_result': search_result
            }
            
            print(f"   📊 دقة الاختبار: {test_accuracy:.4f}")
            
            self.best_models[model_name] = best_model
        
        self.tuning_results = tuning_results
        return tuning_results
    
    def compare_before_after(self, baseline_results, X_test, y_test):
        """مقارنة الأداء قبل وبعد الضبط"""
        print("\n📊 مقارنة الأداء قبل وبعد ضبط المعاملات...")
        
        comparison_data = []
        
        for model_name, tuned_info in self.tuning_results.items():
            # الأداء بعد الضبط
            tuned_accuracy = tuned_info['test_accuracy']
            tuned_model = tuned_info['best_model']
            
            # البحث عن الأداء الأساسي
            baseline_accuracy = None
            for _, row in baseline_results.iterrows():
                if row['Model'] == model_name:
                    baseline_accuracy = row['Accuracy']
                    break
            
            if baseline_accuracy is not None:
                improvement = tuned_accuracy - baseline_accuracy
                comparison_data.append({
                    'Model': model_name,
                    'Baseline Accuracy': baseline_accuracy,
                    'Tuned Accuracy': tuned_accuracy,
                    'Improvement': improvement
                })
                
                print(f"   {model_name}:")
                print(f"      قبل: {baseline_accuracy:.4f}")
                print(f"      بعد: {tuned_accuracy:.4f}")
                print(f"      تحسن: {improvement:+.4f}")
        
        # رسم المقارنة
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            plt.figure(figsize=(12, 6))
            
            x = np.arange(len(comparison_df))
            width = 0.35
            
            plt.bar(x - width/2, comparison_df['Baseline Accuracy'], width, 
                   label='قبل الضبط', alpha=0.7, color='skyblue')
            plt.bar(x + width/2, comparison_df['Tuned Accuracy'], width, 
                   label='بعد الضبط', alpha=0.7, color='lightcoral')
            
            plt.xlabel('النماذج')
            plt.ylabel('الدقة')
            plt.title('مقارنة الدقة قبل وبعد ضبط المعاملات')
            plt.xticks(x, comparison_df['Model'], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            return comparison_df
        
        return None
    
    def plot_hyperparameter_importance(self):
        """رسم أهمية المعاملات للنماذج"""
        print("\n📈 تحليل أهمية المعاملات...")
        
        for model_name, tuned_info in self.tuning_results.items():
            search_result = tuned_info['search_result']
            best_params = tuned_info['best_params']
            
            print(f"\n🎯 أهم المعاملات لـ {model_name}:")
            for param, value in best_params.items():
                print(f"   {param}: {value}")
            
            # لـ Random Forest يمكننا رسم أهمية المعاملات
            if model_name == 'Random Forest':
                best_model = tuned_info['best_model']
                
                # أهمية المعاملات من نتائج البحث
                if hasattr(search_result, 'cv_results_'):
                    results_df = pd.DataFrame(search_result.cv_results_)
                    
                    # استخراج أهم المعاملات المؤثرة
                    param_columns = [col for col in results_df.columns if col.startswith('param_')]
                    mean_test_scores = results_df['mean_test_score']
                    
                    # تحليل تأثير المعاملات
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    axes = axes.flatten()
                    
                    for i, param in enumerate(param_columns[:4]):
                        if i < len(axes):
                            param_name = param.replace('param_', '')
                            param_values = results_df[param].astype(str)
                            
                            # تجميع النتائج حسب قيمة المعامل
                            param_analysis = results_df.groupby(param)['mean_test_score'].agg(['mean', 'std'])
                            
                            axes[i].bar(range(len(param_analysis)), param_analysis['mean'], 
                                      yerr=param_analysis['std'], capsize=5, color='lightgreen')
                            axes[i].set_title(f'تأثير {param_name} على الدقة')
                            axes[i].set_xlabel(param_name)
                            axes[i].set_ylabel('متوسط الدقة')
                            axes[i].set_xticks(range(len(param_analysis)))
                            axes[i].set_xticklabels(param_analysis.index, rotation=45)
                    
                    plt.tight_layout()
                    plt.show()
    
    def save_tuned_models(self, models_path='models/'):
        """حفظ النماذج المظبوطة"""
        import os
        os.makedirs(models_path, exist_ok=True)
        
        print(f"💾 حفظ النماذج المظبوطة...")
        
        for model_name, model in self.best_models.items():
            filename = f"tuned_{model_name.lower().replace(' ', '_')}.pkl"
            filepath = os.path.join(models_path, filename)
            
            joblib.dump(model, filepath)
            print(f"   ✅ تم حفظ {model_name} المظبوط في {filename}")
        
        # حفظ أفضل نموذج مظبوط
        best_tuned_model_name = max(self.tuning_results.items(), 
                                  key=lambda x: x[1]['test_accuracy'])[0]
        best_tuned_model = self.best_models[best_tuned_model_name]
        
        best_model_path = os.path.join(models_path, 'best_tuned_model.pkl')
        joblib.dump(best_tuned_model, best_model_path)
        print(f"   🏆 تم حفظ أفضل نموذج مظبوط في best_tuned_model.pkl")
    
    def complete_tuning_pipeline(self, df, baseline_results, target_col='target', test_size=0.2):
        """خطة ضبط كاملة"""
        print("🎯 بدء خطة الضبط الكاملة للمعاملات...")
        
        from sklearn.model_selection import train_test_split
        
        # تحضير البيانات
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # ضبط المعاملات
        tuning_results = self.tune_models(X_train, X_test, y_train, y_test)
        
        # مقارنة النتائج
        comparison_df = self.compare_before_after(baseline_results, X_test, y_test)
        
        # تحليل المعاملات
        self.plot_hyperparameter_importance()
        
        # حفظ النماذج
        self.save_tuned_models()
        
        print("\n🎉 تم الانتهاء من ضبط المعاملات بنجاح!")
        return tuning_results, comparison_df

# دالة مساعدة
def perform_hyperparameter_tuning(df, baseline_results, target_col='target'):
    tuner = HyperparameterTuner()
    tuning_results, comparison_df = tuner.complete_tuning_pipeline(df, baseline_results, target_col)
    return tuning_results, comparison_df, tuner