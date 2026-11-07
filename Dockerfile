FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY holo ./holo
COPY scripts ./scripts
COPY models ./models

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

# docker run --rm -v <папка с 4 PNG>:/in -v <папка результата>:/out holo-doe
ENTRYPOINT ["python", "scripts/synthesize.py", "/in", "--out", "/out"]
