FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r ./app/requirements.txt

EXPOSE 7000 

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]