import argparse
import logging
import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Пути к файлам и общие данные
pd.set_option('future.no_silent_downcasting', True)
best_features = ['CryoSleep', 'RoomService', 'Spa', 'VRDeck', 'Deck', 'Side', 'SumSpends']
model_params_path = 'data/model/model_params.pkl'
predictions_path = 'data/results.csv'
logs_path = 'data/log_file.log'


# Обертка для модели
class My_Classifier_Model:
    def __init__(self):
        self.model = None
        logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(logs_path)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        self.logger = logger

    @staticmethod
    def train(path_to_dataset, logger):
        logger.info("Training the model...")

        train_df = pd.read_csv(f'{path_to_dataset}', index_col='PassengerId')

        train_df['Transported'].replace({False: 0, True: 1})
        train_df[['Deck', 'CabinNumber', 'Side']] = train_df['Cabin'].str.split('/', expand=True)
        train_df.drop(['Cabin', 'Name'], axis=1, inplace=True)

        object_columns = [column for column in train_df.columns if
                          train_df[column].dtype == 'object' or train_df[column].dtype == 'category']
        expense_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        train_df['SumSpends'] = train_df[expense_columns].sum(axis=1)

        null_columns = train_df.isnull().sum().sort_values(ascending=False)
        null_columns = list(null_columns[null_columns > 1].index)

        oc = OrdinalEncoder()
        train_df[object_columns] = train_df[object_columns].astype('category')
        train_df[object_columns] = oc.fit_transform(train_df[object_columns])
        ct = ColumnTransformer([("imp", SimpleImputer(strategy='mean'), null_columns)])
        train_df[null_columns] = ct.fit_transform(train_df[null_columns])

        X = train_df
        y = X.pop('Transported')

        model = CatBoostClassifier(verbose=False, eval_metric='Accuracy', iterations=450,
                                   learning_rate=0.050891473714946345, depth=7)
        model.fit(X[best_features], y)

        logger.info("Training completed.")

        joblib.dump(model, model_params_path)

        logger.info("Model saved.")

    @staticmethod
    def predict(path_to_dataset, logger):
        if os.path.exists(model_params_path):
            model = joblib.load(model_params_path)
            logger.info("Predicting with the model...")
        else:
            logger.error("Model not trained. Please run 'train' command first.")
            return

        test_df = pd.read_csv(f'{path_to_dataset}', index_col='PassengerId')

        test_df[['Deck', 'Num', 'Side']] = test_df['Cabin'].str.split('/', expand=True)
        test_df.drop(['Cabin', 'Name'], axis=1, inplace=True)

        object_columns = [column for column in test_df.columns if
                          test_df[column].dtype == 'object' or test_df[column].dtype == 'category']
        expense_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        test_df['SumSpends'] = test_df[expense_columns].sum(axis=1)
        null_columns = test_df.isnull().sum().sort_values(ascending=False)
        null_columns = list(null_columns[null_columns > 1].index)

        oc = OrdinalEncoder()
        test_df[object_columns] = test_df[object_columns].astype('category')
        test_df[object_columns] = oc.fit_transform(test_df[object_columns])
        ct = ColumnTransformer([("imp", SimpleImputer(strategy='mean'), null_columns)])
        test_df[null_columns] = ct.fit_transform(test_df[null_columns])

        prediction = model.predict(test_df[best_features])

        logger.info("Prediction completed.")

        final = pd.DataFrame()
        final.index = test_df.index
        final['Transported'] = prediction
        final['Transported'].replace(0, False, inplace=True)
        final['Transported'].replace(1, True, inplace=True)
        final.to_csv(predictions_path)

        logger.info("Prediction saved.")


if __name__ == '__main__':
    # Cоздание папок для логов и параметров модели
    os.makedirs('data', exist_ok=True)
    os.makedirs('data\\model', exist_ok=True)

    model = My_Classifier_Model()

    # Настройка парсера для обработки поведения модели через CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--dataset')

    args = parser.parse_args()

    if args.command == 'train':
        if args.dataset:
            model.train(args.dataset, model.logger)
        else:
            model.logger.error("Path is required.")
    elif args.command == 'predict':
        if args.dataset:
            model.predict(args.dataset, model.logger)
        else:
            model.logger.error("Path is required.")
    else:
        model.logger.error(f"Unknown command: {args.command}.")
