# Stage 1: Build
FROM python:3.11.11-bookworm AS build

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv
RUN python3 -m venv .venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

COPY ./requirements-prod.txt requirements.txt

RUN pip install --no-cache-dir -r ./requirements.txt

# Stage 2: Run
FROM python:3.11.11-bookworm

WORKDIR /app

COPY --from=build /app/.venv ./.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

COPY . /app

# Don't produce .pyc files
ENV PYTHONUNBUFFERED=1
# Don't buffer stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "gradio-dashboard.py"]

EXPOSE 7860
