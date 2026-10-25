FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
# Run as an unprivileged user: a bug in the app then cannot write outside /srv/data.
RUN useradd --system --uid 10001 gateway && mkdir -p /srv/data && chown gateway /srv/data
USER gateway
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
