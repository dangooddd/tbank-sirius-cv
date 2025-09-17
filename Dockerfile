FROM rockylinux:9-minimal

RUN microdnf install -y python-pip curl jq
RUN pip install uv

COPY uv.lock pyproject.toml LICENSE start.sh /app/
COPY src/ /app/src/

WORKDIR /app

RUN chmod +x start.sh
RUN uv venv --python 3.11 .venv
RUN uv sync --no-dev

EXPOSE 8000

CMD ["./start.sh"]
