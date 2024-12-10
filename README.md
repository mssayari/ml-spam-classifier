# Phase 1 Report: Dataset Generation for Email Classification

## Project Overview

In this phase, we aimed to build a Python-based pipeline to generate a labeled dataset of emails. The primary objective
was to utilize the OpenAI API to generate a dataset of 2,000 emails, evenly labeled as “spam” and “non-spam.” This
dataset will serve as the foundation for future email classification tasks.

## Key Features of the Code

The code is structured around the DatasetHandler class, which handles dataset generation, storage, and management.
Below, we break down the main features:

### 1. Dataset Generation

The DatasetHandler class initializes with parameters such as the API key, the number of samples, batch size, and file
paths. It also ensures the specified output directory exists.


### 2. Batch Email Generation

The `generate_emails_batch` function sends a prompt to the OpenAI API, requesting a batch of emails. Each email is assigned a random label (“spam” or “non-spam”) and contains 2-3 sentences. The function then processes the API response and returns a list of email objects.

### 3. Data Storage
The `append_to_file` function handles saving the generated emails to a JSON file. It ensures that new data is appended to the file if it already exists, or creates a new file if not. This ensures no data is lost, even if the process is interrupted.

### 4. Dataset Creation
The `create_dataset` function orchestrates the entire dataset generation process. It breaks the dataset into batches, generates the specified number of emails, and appends them to the file. If there are any remaining emails to generate (less than a full batch), it handles them at the end.

### Example Usage

```python
if __name__ == "__main__":
    dataset_handler = DatasetHandler(os.getenv("OPEN_AI_API_KEY"), 'data', 'dataset.json', 2000, 25)
    dataset_handler.create_dataset()
```
## Progress Achieved

### Completed Tasks:
- Implemented the `DatasetHandler` class to generate and store labeled emails.
- Successfully generated and saved a dataset of 2,000 emails using OpenAI’s API.
- Implemented error handling for robust dataset creation.

### Output Example:

```json
[
    {
        "label": "spam",
        "text": "Congratulations! You've won a free trip. Click here to claim your prize now."
    },
    {
        "label": "non-spam",
        "text": "Hi John, can we reschedule our meeting for next week?"
    }
]
```

## Next Steps

In the upcoming phase, we plan to focus on the following tasks:
- Validate the dataset for consistency and quality.
- lore potential preprocessing tasks for the dataset.
- n for phase two, which involves training and evaluating machine learning models using this dataset.


