# Project Report: Email Classification

## Project Overview

This project focuses on developing a comprehensive pipeline for email classification. The aim is to generate a
high-quality labeled dataset of emails and use it to train and evaluate machine learning models. The project consists of
two major phases: **Dataset Generation** and **Model Training**. This report documents the progress, methodologies, and
outcomes of the first phase (Dataset Generation) and sets the foundation for subsequent phases.

---

## Phase 1: Dataset Generation

The primary goal of this phase was to build a Python-based pipeline to generate a labeled dataset of emails. This
dataset serves as the foundation for the classification tasks in the next phase.

### Key Components

#### 1. DatasetHandler Class

The core of the pipeline is the `DatasetHandler` class, which manages dataset generation, storage, and management. It is
designed with modularity and robustness in mind to ensure ease of future scalability.

#### 2. Email Generation

The `generate_emails_batch` function leverages AI APIs to generate email samples. Each email is randomly labeled as
“spam” or “non-spam” and contains concise content (2-3 sentences).

#### 3. Data Storage

The `append_to_file` function handles saving emails in a JSON format. It ensures data integrity by appending new data to
existing files or creating new ones when necessary.

#### 4. Dataset Creation Workflow

The `create_dataset` function orchestrates the email generation process in batches. It ensures that the exact number of
required samples is generated efficiently, even handling residual emails if they do not fit into a full batch.

### Example Usage of the Code

```python
if __name__ == "__main__":
    dataset_handler = DatasetHandler(os.getenv("OPEN_AI_API_KEY"), 'data', 'dataset.json', 20000, 50)
    dataset_handler.create_dataset()
```
---

### Dataset Expansion

To enhance the dataset for better performance during model training, we expanded the initial dataset of 20,000 records to
30,000 records. This was achieved by incorporating additional records generated using different AI tools:

1. **OpenAI API:** Initially generated 20,000 labeled email samples.
2. **Claude.ai:** Used a custom prompt to generate 5,000 additional records with diverse content.
3. **Gemini:** Created 5,000 records using a slightly modified prompt to ensure variation.

#### Prompts Used for Email Generation

- **Claude.ai Prompt:**

```text
"Generate diverse email records labeled as 'spam' or 'non-spam.' Include 2-3 sentences per email. Spam emails should mimic marketing content or scams, while non-spam emails should resemble personal or professional communication. Ensure unique and varied content for each email."
```

- **Gemini Prompt:**

```text
  "Create labeled email records, equally divided into 'spam' and 'non-spam.' Spam emails should focus on promotions, scams, or phishing attempts, and non-spam emails should include everyday messages like meeting notes or family updates. Keep the content concise (2-3 sentences) and avoid repetition."
```

---

### Achievements in Phase 1

- **Pipeline Development:** Successfully implemented the `DatasetHandler` class to handle dataset creation and
  management.
- **Initial Dataset Generation:** Created a labeled dataset of 20,000 emails using the OpenAI API.
- **Dataset Expansion:** Expanded the dataset to 30,000 emails using Claude.ai and Gemini, ensuring diversity and
  robustness.

#### Sample Output

```json
[
  {
    "label": "spam",
    "text": "Congratulations! You've won a free trip. Click here to claim your prize now. Hurry, this offer won't last long!"
  },
  {
    "label": "non-spam",
    "text": "Hi John, can we reschedule our meeting for next week? I have a conflict on Monday."
  }
]
```

### Next Step

## Phase 2: Model Training and Evaluation

In the next phase, we will utilize the generated dataset to develop and evaluate machine learning models for email
classification. Key activities will include:

- Preprocessing the dataset for consistency and quality.
- Selecting and training machine learning models.
- Evaluating model performance using standard metrics.
- Iterating on the model to achieve optimal results.

This document will be updated as the project progresses, including results and insights from subsequent phases.