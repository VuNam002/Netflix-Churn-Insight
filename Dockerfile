# Serving image: the FastAPI app plus the newest trained model.
# Build after `kedro run`:  docker build -t netflix-churn .
# Run:                      docker run -p 8000:8000 netflix-churn
FROM python:3.11-slim

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY src ./src
COPY conf/base ./conf/base
# Only the selected model (classifier.pkl) is needed at serving time: not the data,
# and not the other models in data/06_models/all_models.
COPY data/06_models/classifier.pkl ./data/06_models/classifier.pkl

# The app finds conf/ and data/ relative to src/, so run it from the source tree
# instead of installing the package.
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "netflix.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
