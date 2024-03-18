FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install spaceshiptitanic-0.1.0-py3-none-any.whl

CMD [ "flask", "run" ]
