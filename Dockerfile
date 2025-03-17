FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4
RUN python -m nltk.downloader punkt_tab

COPY . .

ENV FLASK_APP=app.py

RUN pip install gunicorn

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl --fail http://0.0.0.0:8080/ || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
