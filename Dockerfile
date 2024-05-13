FROM python:3.10
WORKDIR /app
RUN pip install --upgrade pip && \
    pip install poetry

COPY pyproject.toml poetry.lock ./
COPY . ./

RUN poetry install --no-root
EXPOSE 8000
ENTRYPOINT [ "poetry" ,"run", "uvicorn", "main:app", "--host", "0.0.0.0" ]
