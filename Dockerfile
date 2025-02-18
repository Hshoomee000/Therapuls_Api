FROM python:3.10

# تثبيت المكتبات المطلوبة لـ OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# تحديد مسار العمل
WORKDIR /app

# نسخ الملفات الضرورية
COPY requirements.txt /app/

# تثبيت الحزم المطلوبة
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع
COPY . /app

# ضبط متغير البيئة PORT
ENV PORT=8000

# تشغيل التطبيق بشكل صحيح مع تفسير متغير البيئة
CMD ["sh", "-c", "uvicorn hello:app --host 0.0.0.0 --port ${PORT}"]
