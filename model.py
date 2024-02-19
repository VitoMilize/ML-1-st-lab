import argparse
import logging
import os
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer

binary_features = ['VIP', 'CryoSleep']
cryo_sleep_depending_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features_without_cabin = ['HomePlanet', 'Destination']
categorical_features_cleared = ['HomePlanet', 'Destination', 'Deck', 'Side']
numerical_features = ['ShoppingMall', 'Spa', 'VRDeck', 'RoomService', 'FoodCourt', 'Age']
imputer = KNNImputer(n_neighbors=1)


class My_Classifier_Model:
    def __init__(self):
        self.model = None

    def train(self, path_to_dataset):
        logging.info("Training the model")

        df = pd.read_csv(f'{path_to_dataset}')
        df = df.drop(columns=['Name', 'PassengerId'])

        for i in range(len(df)):
            if df.at[i, 'CryoSleep'] == True:
                for feature in cryo_sleep_depending_features:
                    df.at[i, feature] = 0.0

        rows_with_many_missing_values = df[df.isna().sum(axis=1) > 1]
        rows_with_one_missing_value = df[df.isna().sum(axis=1) == 1]
        df = df.dropna()
        df[["Deck", "CabinNumber", "Side"]] = df["Cabin"].str.split("/", expand=True)
        df = df.drop(columns=['Cabin', 'CabinNumber'])
        df = df.astype({'CryoSleep': int, 'VIP': int, 'Transported': int})
        df = pd.get_dummies(df, columns=categorical_features_cleared)

        for feature in numerical_features:
            rows_with_missing_feature = rows_with_one_missing_value[rows_with_one_missing_value[feature].isna()]
            rows_with_missing_feature[["Deck", "CabinNumber", "Side"]] = rows_with_missing_feature["Cabin"].str.split(
                "/", expand=True)
            rows_with_missing_feature = rows_with_missing_feature.drop(columns=['Cabin', 'CabinNumber'])
            rows_with_missing_feature = rows_with_missing_feature.astype(
                {'CryoSleep': int, 'VIP': int, 'Transported': int})
            rows_with_missing_feature = pd.get_dummies(rows_with_missing_feature, columns=categorical_features_cleared)
            df = pd.concat([df, rows_with_missing_feature], ignore_index=True)
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        for feature in categorical_features_without_cabin:
            rows_with_missing_feature = rows_with_one_missing_value[rows_with_one_missing_value[feature].isna()]
            rows_with_missing_feature[["Deck", "CabinNumber", "Side"]] = rows_with_missing_feature["Cabin"].str.split(
                "/", expand=True)
            rows_with_missing_feature = rows_with_missing_feature.drop(columns=['Cabin', 'CabinNumber'])
            rows_with_missing_feature = rows_with_missing_feature.astype(
                {'CryoSleep': int, 'VIP': int, 'Transported': int})
            rows_with_missing_feature = pd.get_dummies(rows_with_missing_feature, columns=categorical_features_cleared)
            df = pd.concat([df, rows_with_missing_feature], ignore_index=True)
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        for feature in binary_features:
            rows_with_missing_feature = rows_with_one_missing_value[rows_with_one_missing_value[feature].isna()]
            rows_with_missing_feature[["Deck", "CabinNumber", "Side"]] = rows_with_missing_feature["Cabin"].str.split(
                "/", expand=True)
            rows_with_missing_feature = rows_with_missing_feature.drop(columns=['Cabin', 'CabinNumber'])
            rows_with_missing_feature = pd.get_dummies(rows_with_missing_feature, columns=categorical_features_cleared)
            df = pd.concat([df, rows_with_missing_feature], ignore_index=True)
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        rows_with_missing_feature = rows_with_one_missing_value[rows_with_one_missing_value['Cabin'].isna()]
        rows_with_missing_feature = rows_with_missing_feature.drop(columns='Cabin')
        rows_with_missing_feature = rows_with_missing_feature.astype({'CryoSleep': int, 'VIP': int, 'Transported': int})
        rows_with_missing_feature = pd.get_dummies(rows_with_missing_feature,
                                                   columns=categorical_features_without_cabin)
        df = pd.concat([df, rows_with_missing_feature], ignore_index=True)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        rows_with_many_missing_values = rows_with_many_missing_values.drop(columns=['Cabin', 'HomePlanet', 'Destination'])

        df = pd.concat([df, rows_with_many_missing_values], ignore_index=True)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        target = 'Transported'
        X = df.drop(columns=target)
        y = df[target]

        model = CatBoostClassifier(iterations=494, learning_rate=0.08537674565386481, depth=4, random_state=42)
        model.fit(X, y)

        logging.info("Training completed.")

        joblib.dump(model, 'model_params.pkl')

        logging.info("Model saved.")

    def predict(self, dataset):
        logging.info("Predicting with the model")

        model_params_path = 'model_params.pkl'

        if os.path.exists(model_params_path):
            model = joblib.load(model_params_path)
        else:
            logging.error("Error: Model not trained. Please run 'train' command first.")
            return

        df = pd.read_csv(f'{dataset}')
        df = df.drop(columns=['Name'])

        for i in range(len(df)):
            if df.at[i, 'CryoSleep'] == True:
                for feature in cryo_sleep_depending_features:
                    df.at[i, feature] = 0.0

        rows_with_many_missing_values = df[df.isna().sum(axis=1) > 1]
        rows_with_one_missing_value = df[df.isna().sum(axis=1) == 1]
        df = df.dropna()
        df[["Deck", "CabinNumber", "Side"]] = df["Cabin"].str.split("/", expand=True)
        df = df.drop(columns=['Cabin', 'CabinNumber'])
        df = df.astype({'CryoSleep': int, 'VIP': int})
        df = pd.get_dummies(df, columns=categorical_features_cleared)

        for feature in numerical_features:
            rows_with_missing_feature = rows_with_one_missing_value[rows_with_one_missing_value[feature].isna()]
            rows_with_missing_feature[["Deck", "CabinNumber", "Side"]] = rows_with_missing_feature["Cabin"].str.split(
                "/", expand=True)
            rows_with_missing_feature = rows_with_missing_feature.drop(columns=['Cabin', 'CabinNumber'])
            rows_with_missing_feature = rows_with_missing_feature.astype({'CryoSleep': int, 'VIP': int})
            rows_with_missing_feature = pd.get_dummies(rows_with_missing_feature, columns=categorical_features_cleared)
            df = pd.concat([df, rows_with_missing_feature], ignore_index=True)
            ids = df['PassengerId']
            df = df.drop(columns=['PassengerId'])
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            df['PassengerId'] = ids

        for feature in categorical_features_without_cabin:
            rows_with_missing_feature = rows_with_one_missing_value[rows_with_one_missing_value[feature].isna()]
            rows_with_missing_feature[["Deck", "CabinNumber", "Side"]] = rows_with_missing_feature["Cabin"].str.split(
                "/", expand=True)
            rows_with_missing_feature = rows_with_missing_feature.drop(columns=['Cabin', 'CabinNumber'])
            rows_with_missing_feature = rows_with_missing_feature.astype({'CryoSleep': int, 'VIP': int})
            rows_with_missing_feature = pd.get_dummies(rows_with_missing_feature, columns=categorical_features_cleared)
            df = pd.concat([df, rows_with_missing_feature], ignore_index=True)
            ids = df['PassengerId']
            df = df.drop(columns=['PassengerId'])
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            df['PassengerId'] = ids

        for feature in binary_features:
            rows_with_missing_feature = rows_with_one_missing_value[rows_with_one_missing_value[feature].isna()]
            rows_with_missing_feature[["Deck", "CabinNumber", "Side"]] = rows_with_missing_feature["Cabin"].str.split(
                "/", expand=True)
            rows_with_missing_feature = rows_with_missing_feature.drop(columns=['Cabin', 'CabinNumber'])
            rows_with_missing_feature = pd.get_dummies(rows_with_missing_feature, columns=categorical_features_cleared)
            df = pd.concat([df, rows_with_missing_feature], ignore_index=True)
            ids = df['PassengerId']
            df = df.drop(columns=['PassengerId'])
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            df['PassengerId'] = ids

        rows_with_missing_feature = rows_with_one_missing_value[rows_with_one_missing_value['Cabin'].isna()]
        rows_with_missing_feature = rows_with_missing_feature.drop(columns='Cabin')
        rows_with_missing_feature = rows_with_missing_feature.astype({'CryoSleep': int, 'VIP': int})
        rows_with_missing_feature = pd.get_dummies(rows_with_missing_feature,
                                                   columns=categorical_features_without_cabin)
        df = pd.concat([df, rows_with_missing_feature], ignore_index=True)
        ids = df['PassengerId']
        df = df.drop(columns=['PassengerId'])
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        df['PassengerId'] = ids

        rows_with_many_missing_values = rows_with_many_missing_values.drop(
            columns=['Cabin', 'HomePlanet', 'Destination'])

        df = pd.concat([df, rows_with_many_missing_values], ignore_index=True).reset_index(drop=True)
        ids = df['PassengerId']
        df = df.drop(columns=['PassengerId'])
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        predictions = model.predict(df).astype(bool)

        logging.info("Prediction completed.")

        output = pd.DataFrame({'PassengerId': ids, 'Transported': predictions})
        output.to_csv('predictions.csv', index=False)

        logging.info("Prediction saved.")


if __name__ == '__main__':
    model = My_Classifier_Model()

    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--dataset')

    args = parser.parse_args()

    if args.command == 'train':
        if args.dataset:
            model.train(args.dataset)
        else:
            print("Path is required")
    elif args.command == 'predict':
        if args.dataset:
            model.predict(args.dataset)
        else:
            print("Path is required")
    else:
        print(f"Unknown command: {args.command}")
