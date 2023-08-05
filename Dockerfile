FROM python:3.10.9

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN pip install .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
