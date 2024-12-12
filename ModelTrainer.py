import os
from dotenv import load_dotenv
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load environment variables from .env file
load_dotenv()

class ModelTrainer:
    def __init__(self, dataset_path, models_path, results_path, results_file="results.json"):
        """
        Initialize the ModelTrainer.

        :param dataset_path: Path to the dataset JSON file.
        :param models_path: Path to save trained models.
        :param results_path: Path to save evaluation results.
        :param results_file: File to save evaluation results.
        """
        self.dataset_path = dataset_path
        self.models_path = models_path
        self.results_path = results_path
        self.results_file = results_file
        self.models = {}
        self.results = {}

        # Create directories if they don't exist
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)

    def load_dataset(self):
        """
        Load the dataset from a JSON file and preprocess it.
        """
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['label'] = df['label'].map({'spam': 1, 'non-spam': 0})
        self.X = df['email']
        self.y = df['label']

    def preprocess(self):
        """
        Split the dataset into training and testing sets and vectorize the text.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.vectorizer = CountVectorizer()
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)

    def train_naive_bayes(self):
        """
        Train and evaluate a Naive Bayes model.
        """
        model = MultinomialNB()
        model.fit(self.X_train_vec, self.y_train)
        self.evaluate_model(model, "Naive Bayes")

    def train_random_forest(self):
        """
        Train and evaluate a Random Forest model.
        """
        model = RandomForestClassifier(random_state=42)
        model.fit(self.X_train_vec, self.y_train)
        self.evaluate_model(model, "Random Forest")

    def train_svm(self):
        """
        Train and evaluate an SVM model.
        """
        model = SVC()
        model.fit(self.X_train_vec, self.y_train)
        self.evaluate_model(model, "SVM")

    def train_logistic_regression(self):
        """
        Train and evaluate a Logistic Regression model.
        """
        model = LogisticRegression()
        model.fit(self.X_train_vec, self.y_train)
        self.evaluate_model(model, "Logistic Regression")

    def train_bert(self):
        """
        Train and evaluate a BERT model using TensorFlow.
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        train_encodings = tokenizer(list(self.X_train), truncation=True, padding=True, max_length=128)
        test_encodings = tokenizer(list(self.X_test), truncation=True, padding=True, max_length=128)

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            self.y_train.values
        )).batch(16)

        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            self.y_test.values
        )).batch(16)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        )

        model.fit(train_dataset, epochs=3)

        predictions = model.predict(test_dataset)
        y_pred = tf.argmax(predictions.logits, axis=1).numpy()

        self.results["BERT"] = {
            "Precision": precision_score(self.y_test, y_pred),
            "Recall": recall_score(self.y_test, y_pred),
            "F1-Score": f1_score(self.y_test, y_pred),
        }

        # Save the model
        model.save_pretrained(self.models_path + "/BERT")

    def evaluate_model(self, model, model_name):
        """
        Evaluate a trained model and store results.

        :param model: The trained model.
        :param model_name: Name of the model.
        """
        y_pred = model.predict(self.X_test_vec)
        self.results[model_name] = {
            "Precision": precision_score(self.y_test, y_pred),
            "Recall": recall_score(self.y_test, y_pred),
            "F1-Score": f1_score(self.y_test, y_pred),
        }
        # Save the model
        joblib.dump(model, f"{self.models_path}/{model_name}.joblib")

    def save_results(self):
        """
        Save the evaluation results to a JSON file.
        """
        with open(self.results_path + "/" + self.results_file, "w") as f:
            json.dump(self.results, f, indent=4)

    def train_and_evaluate(self):
        """
        Train and evaluate all models.
        """
        self.load_dataset()
        self.preprocess()
        self.train_naive_bayes()
        self.train_random_forest()
        self.train_svm()
        self.train_logistic_regression()
        self.train_bert()
        self.save_results()
        print(f"Results saved to {self.results_path}/{self.results_file}")



# Example usage
if __name__ == "__main__":
    trainer = ModelTrainer(
        dataset_path=os.getenv("DATA_DIR") + "/ai_dataset.json",
        models_path=os.getenv("MODELS_DIR"),
        results_path=os.getenv("RESULTS_DIR"),
        results_file="results.json"
    )
    trainer.train_and_evaluate()
