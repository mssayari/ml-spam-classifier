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
import re
import nltk
from sklearn.metrics import confusion_matrix
import seaborn
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load environment variables from .env file
load_dotenv()


class ModelTrainer:
    def __init__(self, dataset_path, models_path, results_path, results_file="results.json"):
        """
        Initialize the ModelTrainer.

        :param dataset_path: Path to the dataset directory.
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

    def load_dataset(self, file_name="emails.json", file_type="json"):
        """
        Load the dataset from a JSON file and preprocess it.
        """

        if file_type == "json":
            with open(self.dataset_path + "/" + file_name, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df['label'] = df['label'].map({'spam': 1, 'non-spam': 0})
            print(df.head())
            self.X = df['email']
            self.y = df['label']
        elif file_type == "csv":
            df = pd.read_csv(self.dataset_path + "/" + file_name)
            df.drop("Unnamed: 0", inplace=True, axis=1)
            df.dropna(inplace=True)
            print(df.head())
            self.X = df['Body']
            self.y = df['Label']

    def preprocess(self):
        """
        Split the dataset into training and testing sets and vectorize the text.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.X_train = self.X_train.apply(self.clean_and_lemmatize)
        self.X_test = self.X_test.apply(self.clean_and_lemmatize)

        self.vectorizer = CountVectorizer()
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)

    def clean_and_lemmatize(self, text):
        """
        Clean and lemmatize the input text.

        :param text: The input text.
        :return: The cleaned and lemmatized text.
        """

        # Convert to string
        text = str(text)

        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z]', ' ', text)

        # Convert to lowercase
        text = text.lower()

        # Tokenize the text
        words = text.split()

        # Remove stopwords
        words = [word for word in words if word not in stopwords.words('english')]

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)

    def train_naive_bayes(self, dataset_name):
        """
        Train and evaluate a Naive Bayes model.
        """
        model = MultinomialNB()
        model.fit(self.X_train_vec, self.y_train)
        self.evaluate_model(model, "Naive Bayes", dataset_name)

    def train_random_forest(self, dataset_name):
        """
        Train and evaluate a Random Forest model.
        """
        model = RandomForestClassifier(random_state=42)
        model.fit(self.X_train_vec, self.y_train)
        self.evaluate_model(model, "Random Forest", dataset_name)

    def train_svm(self, dataset_name):
        """
        Train and evaluate an SVM model.
        """
        model = SVC()
        model.fit(self.X_train_vec, self.y_train)
        self.evaluate_model(model, "SVM", dataset_name)

    def train_logistic_regression(self, dataset_name):
        """
        Train and evaluate a Logistic Regression model.
        """
        model = LogisticRegression()
        model.fit(self.X_train_vec, self.y_train)
        self.evaluate_model(model, "Logistic Regression", dataset_name)

    def train_bert(self, dataset_name):
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
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=5e-5),
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
        model.save_pretrained(self.models_path + f"/{dataset_name}_BERT")

    def evaluate_model(self, model, model_name, dataset_name):
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
        joblib.dump(model, f"{self.models_path}/{dataset_name}_{model_name}.joblib")

        # Plot confusion matrix
        conf = confusion_matrix(y_pred=y_pred, y_true=self.y_test)
        seaborn.heatmap(conf, annot=True, fmt=".1f", linewidths=1.5)
        plt.show()

    def save_results(self, dataset_name):
        """
        Save the evaluation results to a JSON file.
        """
        results_file = f"{dataset_name}_results.json"
        with open(self.results_path + "/" + results_file, "w") as f:
            json.dump(self.results, f, indent=4)

        print(f"Results saved to {self.results_path}/{self.results_file}")

    def start(self):
        """
        Train and evaluate all models.
        """

        # Load and preprocess the first dataset
        # self.load_dataset(file_name="ai_dataset.json", file_type="json")
        # self.preprocess()
        # self.train_and_evaluate_models("ai_dataset")

        # Load and preprocess the second dataset
        # Load and preprocess the second dataset
        self.load_dataset(file_name="completeSpamAssassin.csv", file_type="csv")
        self.preprocess()
        self.train_and_evaluate_models("completeSpamAssassin")

        # Combine datasets
        # self.load_dataset(file_name="ai_dataset.json", file_type="json")
        # df1 = pd.DataFrame({"email": self.X, "label": self.y})
        # self.load_dataset(file_name="completeSpamAssassin.csv", file_type="csv")
        # df2 = pd.DataFrame({"email": self.X, "label": self.y})
        # combined_df = pd.concat([df1, df2], ignore_index=True)
        # self.X = combined_df['email']
        # self.y = combined_df['label']
        # self.preprocess()
        # self.train_and_evaluate_models("combined_dataset")

    def train_and_evaluate_models(self, dataset_name):
        """
        Train and evaluate models on the current dataset.

        :param dataset_name: Name of the dataset being used.
        """
        self.train_naive_bayes(dataset_name)
        self.train_random_forest(dataset_name)
        self.train_svm(dataset_name)
        self.train_logistic_regression(dataset_name)
        self.train_bert(dataset_name)
        print(f"Training and evaluation completed for {dataset_name}")

        # Save the results
        self.save_results(dataset_name)


# Example usage
if __name__ == "__main__":
    trainer = ModelTrainer(
        dataset_path=os.getenv("DATA_DIR"),
        models_path=os.getenv("MODELS_DIR"),
        results_path=os.getenv("RESULTS_DIR"),
        results_file="results.json"
    )
    trainer.start()
