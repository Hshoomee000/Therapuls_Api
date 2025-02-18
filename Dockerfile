FROM python:3.10

# تثبيت المكتبات النظامية اللازمة لـ OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# تحديد مسار العمل
WORKDIR /app

# نسخ ملف requirements.txt
COPY requirements.txt /app/

# تثبيت الحزم من requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي الملفات
COPY . /app

# تحديد المتغيرات البيئية
ENV PORT=8000

# تشغيل التطبيق
CMD ["sh", "-c", "uvicorn hello:app --host 0.0.0.0 --port $PORT"] 
