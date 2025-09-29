# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def load_data(self, file_path):
        print("📂 جاري تحميل البيانات...")
        try:
            df = pd.read_csv(file_path)
            print(f"✅ تم تحميل البيانات - الحجم: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ خطأ في تحميل البيانات: {e}")
            return None
    
    def explore_data(self, df):
        print("\n🔍 التحليل الاستكشافي:")
        print(f"📊 شكل البيانات: {df.shape}")
        print(f"🎯 الأعمدة: {list(df.columns)}")
        print(f"🔢 القيم المفقودة:\n{df.isnull().sum()}")
        return df
    
    def handle_missing_values(self, df):
        print("\n🔄 معالجة القيم المفقودة...")
        missing_before = df.isnull().sum().sum()
        
        if missing_before > 0:
            df_clean = df.copy()
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                df_clean[numeric_cols] = self.imputer.fit_transform(df_clean[numeric_cols])
            
            missing_after = df_clean.isnull().sum().sum()
            print(f"✅ تم معالجة {missing_before - missing_after} قيمة مفقودة")
            return df_clean
        else:
            print("✅ لا توجد قيم مفقودة")
            return df
    
    def encode_categorical(self, df):
        print("\n🔤 ترميز المتغيرات الفئوية...")
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                print(f"   ✅ تم ترميز العمود: {col}")
        
        return df_encoded
    
    def scale_features(self, df, target_col='target'):
        print("\n⚖️ توحيد مقاييس الميزات...")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        df_scaled = pd.concat([X_scaled_df, y], axis=1)
        self.is_fitted = True
        
        print("✅ تم توحيد مقاييس الميزات")
        return df_scaled
    
    def preprocess_pipeline(self, file_path):
        print("🚀 بدء معالجة البيانات...")
        
        df = self.load_data(file_path)
        if df is None:
            return None
        
        self.explore_data(df)
        df_clean = self.handle_missing_values(df)
        df_encoded = self.encode_categorical(df_clean)
        df_final = self.scale_features(df_encoded)
        
        print("🎉 تم الانتهاء من معالجة البيانات!")
        return df_final

def load_and_preprocess_data(file_path):
    processor = DataPreprocessor()
    return processor.preprocess_pipeline(file_path)