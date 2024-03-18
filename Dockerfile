FROM python:3.11-slim

RUN pip install spaceshiptitanic-0.1.0-py3-none-any.whl

WORKDIR /app

COPY . .

CMD [ "flask", "run" ]
