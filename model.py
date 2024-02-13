import argparse
import logging


class My_Classifier_Model:
    def __init__(self):
        self.model = None

    def train(self):
        logging.info("Training the model")
        print('training')
        logging.info("Training completed.")

    def predict(self):
        if self.model is None:
            logging.error("Error: Model not trained. Please run 'train' command first.")
            return

        logging.info("Predicting with the model")
        print('predicting')
        logging.info("Prediction completed.")


if __name__ == '__main__':
    model = My_Classifier_Model()

    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--dataset')

    args = parser.parse_args()

    if args.command == 'train':
        if args.dataset:
            model.train()
        else:
            print("Path is required")
    elif args.command == 'predict':
        if args.dataset:
            model.predict()
        else:
            print("Path is required")
    else:
        print(f"Unknown command: {args.command}")
