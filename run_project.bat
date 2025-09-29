@echo off
chcp 65001
echo.
echo ========================================
echo    🏥 مشروع تشخيص أمراض القلب
echo ========================================
echo.

echo 📊 جاري التحقق من المتطلبات...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ بايثون غير مثبت!
    echo 📥 يرجى تثبيت Python 3.8+ من python.org
    pause
    exit /b 1
)

echo ✅ بايثون مثبت
echo 📦 جاري تثبيت المتطلبات...
pip install -r requirements.txt

echo.
echo 🚀 جاري تشغيل المشروع...
python main.py

echo.
echo ========================================
echo    ✅ تم الانتهاء من التشغيل
echo ========================================
echo.
echo 🌐 لتشغيل التطبيق:
echo    cd app && streamlit run app.py
echo.
pause