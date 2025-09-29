# src/feature_selection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = None
        self.feature_importance_df = None
        
    def random_forest_importance(self, X, y, n_features=10):
        """أهمية الميزات باستخدام Random Forest"""
        print("🌲 حساب أهمية الميزات باستخدام Random Forest...")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        
        # استخراج أهمية الميزات
        importance_scores = rf.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # رسم أهمية الميزات
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance.head(n_features), 
                   x='importance', y='feature', palette='viridis')
        plt.title(f'أهم {n_features} ميزات باستخدام Random Forest')
        plt.xlabel('مستوى الأهمية')
        plt.tight_layout()
        plt.show()
        
        return feature_importance, rf
    
    def recursive_feature_elimination(self, X, y, n_features=8):
        """إزالة الميزات العكسية RFE"""
        print("🔄 تطبيق إزالة الميزات العكسية (RFE)...")
        
        # استخدام الانحدار اللوجستي مع RFE
        estimator = LogisticRegression(random_state=self.random_state, max_iter=1000)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        # استخراج النتائج
        rfe_results = pd.DataFrame({
            'feature': X.columns,
            'rfe_ranking': rfe.ranking_,
            'rfe_selected': rfe.support_
        }).sort_values('rfe_ranking')
        
        print(f"الميزات المختارة بواسطة RFE:")
        selected_features = rfe_results[rfe_results['rfe_selected']]['feature'].tolist()
        for feature in selected_features:
            print(f"  ✅ {feature}")
        
        return rfe_results, rfe
    
    def statistical_feature_selection(self, X, y, k=8):
        """اختيار الميزات باستخدام الأساليب الإحصائية"""
        print("📊 اختيار الميزات باستخدام الأساليب الإحصائية...")
        
        # استخدام ANOVA F-test
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)
        
        # استخراج النتائج
        statistical_scores = pd.DataFrame({
            'feature': X.columns,
            'f_score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('f_score', ascending=False)
        
        # رسم نتائج F-test
        plt.figure(figsize=(12, 8))
        significant_features = statistical_scores[statistical_scores['p_value'] < 0.05]
        
        if len(significant_features) > 0:
            sns.barplot(data=significant_features.head(k), 
                       x='f_score', y='feature', palette='plasma')
            plt.title('أهم الميزات باستخدام ANOVA F-test (p-value < 0.05)')
            plt.xlabel('F-score')
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ لا توجد ميزات ذات دلالة إحصائية (p-value < 0.05)")
        
        return statistical_scores, selector
    
    def compare_feature_sets(self, X, y, methods_results):
        """مقارنة نتائج طرق اختيار الميزات المختلفة"""
        print("\n🔍 مقارنة نتائج طرق اختيار الميزات...")
        
        # تجميع الميزات المختارة من جميع الطرق
        all_selected_features = set()
        feature_scores = {}
        
        for method_name, (results, _) in methods_results.items():
            if method_name == 'random_forest':
                top_features = results.head(10)['feature'].tolist()
            elif method_name == 'rfe':
                top_features = results[results['rfe_selected']]['feature'].tolist()
            elif method_name == 'statistical':
                top_features = results.head(10)['feature'].tolist()
            
            all_selected_features.update(top_features)
            
            # تعيين درجات للميزات
            for i, feature in enumerate(top_features):
                if feature not in feature_scores:
                    feature_scores[feature] = 0
                feature_scores[feature] += (len(top_features) - i)
        
        # ترتيب الميزات حسب مجموع الدرجات
        final_ranking = pd.DataFrame({
            'feature': list(all_selected_features),
            'composite_score': [feature_scores[feature] for feature in all_selected_features]
        }).sort_values('composite_score', ascending=False)
        
        # رسم الميزات النهائية المختارة
        plt.figure(figsize=(12, 8))
        sns.barplot(data=final_ranking.head(12), x='composite_score', y='feature', palette='coolwarm')
        plt.title('الترتيب النهائي للميزات بناءً على الطرق المختلفة')
        plt.xlabel('الدرجة المركبة')
        plt.tight_layout()
        plt.show()
        
        return final_ranking
    
    def select_final_features(self, X, y, top_k=8):
        """اختيار الميزات النهائية باستخدام الطرق المتعددة"""
        print("🎯 بدء عملية اختيار الميزات النهائية...")
        
        # تطبيق جميع الطرق
        rf_importance, rf_model = self.random_forest_importance(X, y)
        rfe_results, rfe_selector = self.recursive_feature_elimination(X, y)
        statistical_results, stats_selector = self.statistical_feature_selection(X, y)
        
        # تجميع النتائج
        methods_results = {
            'random_forest': (rf_importance, rf_model),
            'rfe': (rfe_results, rfe_selector),
            'statistical': (statistical_results, stats_selector)
        }
        
        # مقارنة واختيار الميزات النهائية
        final_ranking = self.compare_feature_sets(X, y, methods_results)
        
        # اختيار أفضل الميزات
        self.selected_features = final_ranking.head(top_k)['feature'].tolist()
        self.feature_importance_df = final_ranking
        
        print(f"\n🎉 الميزات النهائية المختارة ({top_k} ميزات):")
        for i, feature in enumerate(self.selected_features, 1):
            print(f"  {i}. {feature}")
        
        return self.selected_features, methods_results
    
    def get_reduced_dataset(self, df, target_col='target'):
        """إرجاع البيانات مع الميزات المختارة فقط"""
        if self.selected_features is None:
            raise ValueError("يجب استدعاء select_final_features أولاً")
        
        features_to_keep = self.selected_features + [target_col]
        reduced_df = df[features_to_keep]
        
        print(f"✅ تم اختيار {len(self.selected_features)} ميزة من أصل {len(df.columns)-1}")
        print(f"📊 حجم البيانات بعد اختيار الميزات: {reduced_df.shape}")
        
        return reduced_df

# دالة مساعدة
def perform_feature_selection(df, target_col='target', top_k=8):
    selector = FeatureSelector()
    selected_features, methods_results = selector.select_final_features(
        df.drop(columns=[target_col]), 
        df[target_col], 
        top_k
    )
    reduced_df = selector.get_reduced_dataset(df, target_col)
    
    return reduced_df, selected_features, methods_results, selector