# Natural Language Classifier

## Summary

In this project I will build a full AI pipeline, using a publically accessable dataset. I will:
- Ingest and store that data reliably, using a local script which will form the basis for an AWS Lambda with RDS
- Train a model on that data to understand insights
- Create a chatbot interface that uses your trained model
- Add RAG (Retrieval-Augmented Generation) functionality using simple local files

## How to run

1. Clone the repository.
2. Set up and activate a virtual environment.
3. Run 'pip install datasets pytest scikit-learn transformers accelerate torch' (See requirment.txt for more information.)
4. Run 'python3 ingest_layer/ingest_data.py' to load and clean data.
5. Run 'pytest tests/test_ingest_data.py' to check that ingest layer passes tests.
6. Run 'python3 train_layer/train_news_model.py' to train classifier model.
7. Run 'pytest tests/test_news_classifier' to check that NewsClassifier class passes tests.
8. Run 'python3 classifier_layer/simple_interface.py' to predict news topics through the CLI.

### Citation

This project uses the Topic Labeled News Dataset from NewsCatcher via the GitHub repository by Bugara, (2020).

Bugara, A. (2020). topic-labeled-news-dataset [Software]. GitHub. https://github.com/kotartemiy/topic-labeled-news-dataset
