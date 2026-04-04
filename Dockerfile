FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["infer-harness"]
CMD ["synthetic", "--out", "results/docker.jsonl"]
