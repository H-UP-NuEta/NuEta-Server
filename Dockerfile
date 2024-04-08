# Python 3.10.12를 기반 이미지로 사용
FROM python:3.10.12

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 종속성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Poetry 설치
RUN curl -sSL https://install.python-poetry.org | python3 -

# Poetry 환경 변수 설정, 시스템 PATH에 Poetry 바이너리 추가
ENV PATH="${PATH}:/root/.local/bin"

# 애플리케이션의 종속성 파일 복사
COPY pyproject.toml poetry.lock* /app/

# Poetry를 사용하여 종속성 설치, 가상 환경 생성 방지
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# 애플리케이션 코드 복사
COPY . /app

# 애플리케이션 실행
CMD ["unicorn", "main:app"]
