from openai import OpenAI
import os
import json
import random
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class DatasetHandler:
    def __init__(self, api_key, output_dir, output_file, num_samples=2000, batch_size=20):
        """
        Initialize the DatasetHandler.

        :param api_key: Your OpenAI API key.
        :param output_dir: Directory to save the dataset.
        :param output_file: File to save the generated dataset.
        :param num_samples: Total number of email samples to generate.
        :param batch_size: Number of emails to generate per API call.
        """
        self.api_key = api_key
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, output_file)
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key)

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_emails_batch(self, batch_size):
        """
        Generate a batch of emails using OpenAi API.

        :param batch_size: Number of emails to generate in this batch.
        :return: List of dictionaries with "label" and "text" keys.
        """
        labels = ["spam" if random.random() < 0.5 else "non-spam" for _ in range(batch_size)]
        prompt = (
            f"Generate {batch_size} emails in JSON format. Each email should have a 'label' "
            f"(either 'spam' or 'non-spam') and 'text'. Use the following labels: {labels}. "
            "Each email should be unique and contain at least 3 sentences."
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Emails",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "description": "The label of the email",
                                    "type": "string",
                                    "enum": ["spam", "non-spam"]
                                },
                                "email": {
                                    "description": "The email text",
                                    "type": "string"
                                },
                                "additionalProperties": False
                            }
                        }
                    }
                },
                temperature=0.7
            )
            response = response.choices[0].message.content
            return json.loads(response)["emails"]

        except Exception as e:
            print(f"Error generating batch emails: {e}")
            return []

    def append_to_file(self, data):
        """
        Append data to the output JSON file.

        :param data: List of email dictionaries to append.
        """
        if os.path.exists(self.output_file):
            with open(self.output_file, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(data)

        with open(self.output_file, "w") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

    def create_dataset(self):
        """
        Generate the dataset of labeled emails and save it to the JSON file.
        """
        num_batches = self.num_samples // self.batch_size

        for batch in tqdm(range(num_batches), desc="Generating emails"):
            batch_data = self.generate_emails_batch(self.batch_size)
            self.append_to_file(batch_data)

        # Handle remaining samples
        remaining = self.num_samples % self.batch_size
        if remaining > 0:
            print(f"\nGenerating remaining {remaining} emails...")
            batch_data = self.generate_emails_batch(remaining)
            self.append_to_file(batch_data)

        print(f"Dataset saved to {self.output_file}")


# Example usage
if __name__ == "__main__":
    dataset_handler = DatasetHandler(
        os.getenv("OPEN_AI_API_KEY"),
        os.getenv("DATA_DIR"),
        'ai_dataset.json',
        20000,
        50
    )
    dataset_handler.create_dataset()
