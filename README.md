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
    3. Run pip install datasets pytest scikit-learn. (See requirment.txt for more information.)
    4. Run

### Citation

This project uses the Hugging Face emotion dataset by Saravia et al (2018).

    @inproceedings{saravia-etal-2018-carer,
        title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
        author = "Saravia, Elvis  and
        Liu, Hsien-Chi Toby  and
        Huang, Yen-Hao  and
        Wu, Junlin  and
        Chen, Yi-Shin",
        booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
        month = oct # "-" # nov,
        year = "2018",
        address = "Brussels, Belgium",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/D18-1404",
        doi = "10.18653/v1/D18-1404",
        pages = "3687--3697",
        abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
    }
