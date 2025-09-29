# src/eda_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class EDAAnalysis:
    def __init__(self):
        plt.style.use('default')
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
    def plot_target_distribution(self, df, target_col='target'):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        target_counts = df[target_col].value_counts()
        axes[0].pie(target_counts.values, labels=['سليم', 'مريض'], 
                   autopct='%1.1f%%', colors=self.colors)
        axes[0].set_title('توزيع حالات أمراض القلب')
        
        sns.countplot(data=df, x=target_col, ax=axes[1], palette=self.colors)
        axes[1].set_title('توزيع حالات أمراض القلب')
        axes[1].set_xticklabels(['سليم', 'مريض'])
        axes[1].set_xlabel('الحالة')
        axes[1].set_ylabel('عدد الحالات')
        
        plt.tight_layout()
        plt.show()
        
        return target_counts
    
    def plot_correlation_matrix(self, df, target_col='target'):
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                   center=0, fmt='.2f', linewidths=0.5)
        plt.title('مصفوفة الارتباط بين متغيرات البيانات')
        plt.tight_layout()
        plt.show()
        
        target_corr = correlation_matrix[target_col].sort_values(ascending=False)
        print("أعلى الارتباطات مع المتغير المستهدف:")
        for feature, corr in target_corr.items():
            if feature != target_col:
                print(f"  {feature}: {corr:.3f}")
        
        return correlation_matrix
    
    def perform_pca_analysis(self, df, target_col='target'):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        pca = PCA()
        X_pca = pca.fit_transform(X)
        
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_, alpha=0.7, color=self.colors[0])
        axes[0].set_xlabel('المكونات الرئيسية')
        axes[0].set_ylabel('نسبة التباين')
        axes[0].set_title('نسبة التباين لكل مكون رئيسي')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(range(1, len(explained_variance) + 1), explained_variance, 
                    marker='o', color=self.colors[1], linewidth=2)
        axes[1].axhline(y=0.95, color='red', linestyle='--', label='95% تباين')
        axes[1].set_xlabel('عدد المكونات الرئيسية')
        axes[1].set_ylabel('التباين المتراكم')
        axes[1].set_title('التباين المتراكم مقابل عدد المكونات')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        n_components = np.argmax(explained_variance >= 0.95) + 1
        print(f"عدد المكونات الرئيسية اللازمة للاحتفاظ ب 95% من التباين: {n_components}")
        
        return pca, X_pca, explained_variance
    
    def generate_eda_report(self, df, target_col='target'):
        print("=" * 50)
        print("📊 بدء التقرير التحليلي الشامل")
        print("=" * 50)
        
        print("\n1. تحليل المتغير المستهدف:")
        target_dist = self.plot_target_distribution(df, target_col)
        
        print("\n2. تحليل الارتباط:")
        correlation_matrix = self.plot_correlation_matrix(df, target_col)
        
        print("\n3. تحليل المكونات الرئيسية (PCA):")
        pca_model, X_pca, explained_variance = self.perform_pca_analysis(df, target_col)
        
        print("\n" + "=" * 50)
        print("✅ تم الانتهاء من التقرير التحليلي")
        print("=" * 50)
        
        return {
            'target_distribution': target_dist,
            'correlation_matrix': correlation_matrix,
            'pca_model': pca_model
        }

def perform_comprehensive_eda(df, target_col='target'):
    analyzer = EDAAnalysis()
    return analyzer.generate_eda_report(df, target_col)