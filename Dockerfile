# Machine Guard API image.
# Build:  docker build -t machine-guard:v1 .
# Run:    docker run --rm -p 8000:8000 -e API_KEY=$API_KEY machine-guard:v1
#
# LAB NOTE: this Dockerfile has TWO planted mistakes. It builds, but the container does not work.
# Find and fix them in the Docker lab (Day 3, 11:00). Read the error messages: they tell you.

# 1. Start from a small official Python image (Debian "slim", Python 3.11 like the codespace).
FROM python:3.11-slim

# 2. Link the image to your GitHub repo, so the package page shows it. Replace YOUR-GITHUB-USER.
LABEL org.opencontainers.image.source="https://github.com/YOUR-GITHUB-USER/machine-guard"

# 3. Don't write .pyc files; print logs straight away instead of buffering them.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 4. All following commands run inside /app in the image.
WORKDIR /app

# 5. Install the Python packages first. This layer is cached, so code changes rebuild in seconds.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# 6. Copy the code.
COPY machine_guard/ machine_guard/
COPY app/ app/

# 7. Settings the app reads. APP_VERSION can be set at build time: --build-arg APP_VERSION=v2
ARG APP_VERSION=v1
ENV APP_VERSION=${APP_VERSION} \
    MODEL_PATH=models/model.joblib \
    PREDICTION_LOG=logs/predictions.db

# 8. Don't run as root. The app user owns logs/ so it can write the SQLite prediction log.
RUN useradd --create-home appuser && mkdir -p logs && chown appuser logs
USER appuser

# 9. Document the port, then start the server.
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
