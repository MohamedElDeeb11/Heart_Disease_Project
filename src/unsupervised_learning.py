# src/unsupervised_learning.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

class UnsupervisedLearning:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.cluster_models = {}
        self.clustering_results = {}
        
    def prepare_data_for_clustering(self, df, target_col='target'):
        """تحضير البيانات للتجميع"""
        print("📊 تحضير البيانات للتجميع...")
        
        # فصل المتغيرات والمتغير المستهدف
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # توحيد المقاييس
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler
    
    def find_optimal_clusters_kmeans(self, X, max_k=15):
        """إيجاد العدد الأمثل للتجمعات باستخدام K-Means"""
        print("🔍 البحث عن العدد الأمثل للتجمعات (K-Means)...")
        
        wcss = []  # Within-Cluster Sum of Squares
        silhouette_scores = []
        
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            
            wcss.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
        
        # رسم منحنى Elbow
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # منحنى Elbow
        axes[0].plot(k_range, wcss, marker='o', color='#FF6B6B', linewidth=2)
        axes[0].set_xlabel('عدد التجمعات (K)')
        axes[0].set_ylabel('WCSS')
        axes[0].set_title('منحنى Elbow لـ K-Means')
        axes[0].grid(True, alpha=0.3)
        
        # منحنى Silhouette
        axes[1].plot(k_range, silhouette_scores, marker='o', color='#4ECDC4', linewidth=2)
        axes[1].set_xlabel('عدد التجمعات (K)')
        axes[1].set_ylabel('متوسط Silhouette Score')
        axes[1].set_title('منحنى Silhouette لـ K-Means')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # العثور على أفضل K
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        print(f"🎯 العدد الأمثل للتجمعات بناءً على Silhouette Score: {optimal_k_silhouette}")
        
        return optimal_k_silhouette, wcss, silhouette_scores
    
    def apply_kmeans_clustering(self, X, n_clusters=2):
        """تطبيق تجميع K-Means"""
        print(f"🔄 تطبيق K-Means مع {n_clusters} تجمعات...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        
        # حساب Silhouette Score
        silhouette_avg = silhouette_score(X, kmeans_labels)
        print(f"   ✅ Silhouette Score: {silhouette_avg:.4f}")
        
        self.cluster_models['kmeans'] = kmeans
        self.clustering_results['kmeans'] = {
            'model': kmeans,
            'labels': kmeans_labels,
            'silhouette_score': silhouette_avg
        }
        
        return kmeans, kmeans_labels
    
    def plot_dendrogram(self, X, method='ward'):
        """رسم Dendrogram للتجميع الهرمي"""
        print("🌳 رسم Dendrogram للتجميع الهرمي...")
        
        plt.figure(figsize=(12, 8))
        
        # حساب الارتباط
        linked = linkage(X, method=method)
        
        # رسم Dendrogram
        dendrogram(linked, orientation='top', 
                  distance_sort='descending', 
                  show_leaf_counts=True)
        
        plt.title('Dendrogram للتجميع الهرمي')
        plt.xlabel('عينات البيانات')
        plt.ylabel('المسافة')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return linked
    
    def apply_hierarchical_clustering(self, X, n_clusters=2, method='ward'):
        """تطبيق التجميع الهرمي"""
        print(f"🌳 تطبيق التجميع الهرمي مع {n_clusters} تجمعات...")
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, 
            metric='euclidean', 
            linkage=method
        )
        hierarchical_labels = hierarchical.fit_predict(X)
        
        # حساب Silhouette Score
        silhouette_avg = silhouette_score(X, hierarchical_labels)
        print(f"   ✅ Silhouette Score: {silhouette_avg:.4f}")
        
        self.cluster_models['hierarchical'] = hierarchical
        self.clustering_results['hierarchical'] = {
            'model': hierarchical,
            'labels': hierarchical_labels,
            'silhouette_score': silhouette_avg
        }
        
        return hierarchical, hierarchical_labels
    
    def visualize_clusters(self, X, cluster_labels, y_true, algorithm_name):
        """تصور التجمعات باستخدام PCA"""
        print(f"🎨 تصور تجمعات {algorithm_name}...")
        
        # تطبيق PCA للتصور
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # التجميع الحقيقي
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, 
                                 cmap='viridis', alpha=0.7, s=50)
        axes[0].set_title(f'التوزيع الحقيقي - أمراض القلب')
        axes[0].set_xlabel('المكون الرئيسي 1')
        axes[0].set_ylabel('المكون الرئيسي 2')
        plt.colorbar(scatter1, ax=axes[0], label='مرض القلب')
        
        # التجميع المتوقع
        scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                 cmap='plasma', alpha=0.7, s=50)
        axes[1].set_title(f'تجمعات {algorithm_name}')
        axes[1].set_xlabel('المكون الرئيسي 1')
        axes[1].set_ylabel('المكون الرئيسي 2')
        plt.colorbar(scatter2, ax=axes[1], label='التجمع')
        
        plt.tight_layout()
        plt.show()
        
        # مقارنة مع التصنيف الحقيقي
        unique_clusters = np.unique(cluster_labels)
        print(f"\n📊 تحليل تجمعات {algorithm_name}:")
        for cluster in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster)[0]
            true_labels_in_cluster = y_true.iloc[cluster_indices] if hasattr(y_true, 'iloc') else y_true[cluster_indices]
            
            disease_ratio = np.mean(true_labels_in_cluster)
            print(f"   التجمع {cluster}: {len(cluster_indices)} عينة، نسبة المرض: {disease_ratio:.2%}")
    
    def compare_clustering_algorithms(self, X, y):
        """مقارنة خوارزميات التجميع المختلفة"""
        print("🔍 مقارنة أداء خوارزميات التجميع...")
        
        comparison_results = []
        
        for algo_name, results in self.clustering_results.items():
            labels = results['labels']
            silhouette = results['silhouette_score']
            
            # Adjusted Rand Score للمقارنة مع التصنيف الحقيقي
            ari = adjusted_rand_score(y, labels)
            
            comparison_results.append({
                'Algorithm': algo_name,
                'Silhouette Score': silhouette,
                'Adjusted Rand Index': ari,
                'Number of Clusters': len(np.unique(labels))
            })
            
            print(f"\n📊 نتائج {algo_name}:")
            print(f"   Silhouette Score: {silhouette:.4f}")
            print(f"   Adjusted Rand Index: {ari:.4f}")
        
        # رسم المقارنة
        comparison_df = pd.DataFrame(comparison_results)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Silhouette Score
        axes[0].bar(comparison_df['Algorithm'], comparison_df['Silhouette Score'], 
                   color=['#FF6B6B', '#4ECDC4'])
        axes[0].set_title('مقارنة Silhouette Score')
        axes[0].set_ylabel('Silhouette Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Adjusted Rand Index
        axes[1].bar(comparison_df['Algorithm'], comparison_df['Adjusted Rand Index'],
                   color=['#45B7D1', '#96CEB4'])
        axes[1].set_title('مقارنة Adjusted Rand Index')
        axes[1].set_ylabel('Adjusted Rand Index')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def analyze_cluster_characteristics(self, df, cluster_labels, target_col='target'):
        """تحليل خصائص كل تجمع"""
        print("📈 تحليل خصائص التجمعات...")
        
        df_analysis = df.copy()
        df_analysis['Cluster'] = cluster_labels
        
        # إحصائيات كل تجمع
        cluster_stats = df_analysis.groupby('Cluster').agg({
            'age': ['mean', 'std'],
            'chol': ['mean', 'std'],
            'trestbps': ['mean', 'std'],
            'thalach': ['mean', 'std'],
            target_col: ['mean', 'count']
        }).round(2)
        
        print("📊 الإحصائيات الوصفية للتجمعات:")
        print(cluster_stats)
        
        # رسم خصائص التجمعات
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = numeric_features.drop(target_col, errors='ignore')
        
        # اختيار أهم 4 ميزات للعرض
        important_features = numeric_features[:4]
        
        n_features = len(important_features)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(important_features):
            if i < len(axes):
                # مخطط الصندوق لكل تجمع
                data_to_plot = [df_analysis[df_analysis['Cluster'] == cluster][feature] 
                              for cluster in np.unique(cluster_labels)]
                
                axes[i].boxplot(data_to_plot, labels=[f'تجمع {c}' for c in np.unique(cluster_labels)])
                axes[i].set_title(f'توزيع {feature} عبر التجمعات')
                axes[i].set_ylabel(feature)
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return df_analysis, cluster_stats
    
    def complete_unsupervised_pipeline(self, df, target_col='target'):
        """خطة التعلم غير المشرف الكاملة"""
        print("🎯 بدء خطة التعلم غير المشرف...")
        
        # 1. تحضير البيانات
        X_scaled, y, scaler = self.prepare_data_for_clustering(df, target_col)
        
        # 2. إيجاد العدد الأمثل للتجمعات
        optimal_k, wcss, silhouette_scores = self.find_optimal_clusters_kmeans(X_scaled)
        
        # 3. تطبيق K-Means
        print(f"\n🔄 تطبيق K-Means مع {optimal_k} تجمعات...")
        kmeans_model, kmeans_labels = self.apply_kmeans_clustering(X_scaled, optimal_k)
        self.visualize_clusters(X_scaled, kmeans_labels, y, 'K-Means')
        
        # 4. التجميع الهرمي
        print(f"\n🌳 تطبيق التجميع الهرمي...")
        self.plot_dendrogram(X_scaled)
        hierarchical_model, hierarchical_labels = self.apply_hierarchical_clustering(X_scaled, optimal_k)
        self.visualize_clusters(X_scaled, hierarchical_labels, y, 'التجميع الهرمي')
        
        # 5. مقارنة الخوارزميات
        comparison_df = self.compare_clustering_algorithms(X_scaled, y)
        
        # 6. تحليل خصائص التجمعات
        df_kmeans_analysis, kmeans_stats = self.analyze_cluster_characteristics(df, kmeans_labels, target_col)
        
        print("\n🎉 تم الانتهاء من التحليل غير المشرف بنجاح!")
        
        return {
            'kmeans_model': kmeans_model,
            'hierarchical_model': hierarchical_model,
            'comparison_results': comparison_df,
            'cluster_analysis': df_kmeans_analysis,
            'optimal_k': optimal_k
        }

# دالة مساعدة
def perform_unsupervised_learning(df, target_col='target'):
    unsupervised = UnsupervisedLearning()
    results = unsupervised.complete_unsupervised_pipeline(df, target_col)
    return results, unsupervised