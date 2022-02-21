# temp stage
FROM python:3.9 as builder

COPY requirements.txt code /app/

WORKDIR /app
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements.txt

# final stage
FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY --from=builder /app /app
RUN pip install --no-cache /wheels/*

RUN addgroup --gid 1001 --system app && \
    adduser --no-create-home --shell /bin/false --disabled-password --uid 1001 --system --group app

USER app

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "webservice:application"]
