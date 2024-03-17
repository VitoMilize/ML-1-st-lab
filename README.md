# Maslov Victor 972201

# "How To"

## Without Docker
1. Download repository
2. Install spaceshiptitanic-0.1.0-py3-none-any.whl by pip or conda
3. Run command 'flask app.py' from root repo folder

### Train model using CLI
Run command 'python model.py train --dataset=/path/to/dataset' from root repo folder

### Get predictions using CLI
Run command 'python model.py predict --dataset=/path/to/dataset' from root repo folder

### Train model using API
Send request POST to http://127.0.0.1:5000/train with body
{
    "dataset_path": "/path/to/dataset"
}

### Get predictions using API
Send request POST to http://127.0.0.1:5000/predict with body
{
    "dataset_path": "/path/to/dataset"
}

## With Docker
1. Download repository
2. Run command 'docker compose up -d' from root repo folder
3. Run command 'flask app.py' from 'app' folder in container

### Train model using CLI
Run command 'python model.py train --dataset=/path/to/dataset' from 'app' folder in container

### Get predictions using CLI
Run command 'python model.py predict --dataset=/path/to/dataset' from 'app' folder in container

### Train model using API
Send request POST to http://127.0.0.1:5000/train in container with body
{
    "dataset_path": "/path/to/dataset/from/app/folder"
}

### Get predictions using API
Send request POST to http://127.0.0.1:5000/predict in container with body
{
    "dataset_path": "/path/to/dataset/from/app/folder"
}

## Utilizided resources
https://education.yandex.ru/handbook/ml https://catboost.ai/ https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/ https://scikit-learn.org/stable/ https://docs.docker.com/reference/cli/docker/image/build/ https://flask.palletsprojects.com/en/3.0.x/
