FROM rockylinux:9-minimal

RUN microdnf install -y python-pip curl
RUN pip install uv

COPY uv.lock pyproject.toml LICENSE start.sh /app/
COPY src/ /app/src/

WORKDIR /app

RUN chmod +x start.sh
RUN uv venv .venv
RUN source .venv/bin/activate
RUN uv sync

EXPOSE 8000

CMD ["start.sh"]
