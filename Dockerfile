FROM python:3.10

# تثبيت المكتبات النظامية
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# تحديد مسار العمل
WORKDIR /app

# نسخ ملف requirements.txt
COPY requirements.txt /app/

# تثبيت الحزم من requirements.txt
RUN pip install -r requirements.txt

# نسخ باقي الملفات
COPY . /app

# تحديد التطبيق لتشغيله
CMD ["uvicorn", "hello:app", "--host", "0.0.0.0", "--port", "8000"]
