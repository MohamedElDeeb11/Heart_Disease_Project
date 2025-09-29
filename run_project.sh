#!/bin/bash

echo "========================================"
echo "   🏥 مشروع تشخيص أمراض القلب"
echo "========================================"
echo ""

echo "📊 جاري التحقق من المتطلبات..."
if ! command -v python3 &> /dev/null; then
    echo "❌ بايثون غير مثبت!"
    echo "📥 يرجى تثبيت Python 3.8+"
    exit 1
fi

echo "✅ بايثون مثبت"
echo "📦 جاري تثبيت المتطلبات..."
pip3 install -r requirements.txt

echo ""
echo "🚀 جاري تشغيل المشروع..."
python3 main.py

echo ""
echo "========================================"
echo "   ✅ تم الانتهاء من التشغيل"
echo "========================================"
echo ""
echo "🌐 لتشغيل التطبيق:"
echo "   cd app && streamlit run app.py"
echo ""