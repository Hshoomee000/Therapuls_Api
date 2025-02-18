FROM python:3.10

# تثبيت المكتبات النظامية
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# باقي الخطوات
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY . /app
