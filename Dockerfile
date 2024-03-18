FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl

COPY . .

RUN pip install spaceshiptitanic-0.1.0-py3-none-any.whl

EXPOSE 5000

CMD [ "flask", "app.py" ]
