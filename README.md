# DCU - Mechanics of Search (CA6005) - Assignment 1
## Simple Information Retrieval (IR) System with VSM, BM25, and BM25 Language Model 


This repository contains the code for a simple Information Retrieval (IR) system that retrieves relevant documents from a collection based on user queries using different retrieval models. The system is implemented in Python and includes functionalities for data loading, preprocessing, building the inverted index, ranking documents, and evaluating retrieval models. This project was done as part of my Mechanics of Search course module assignment under Dublin City University (DCU).

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Dataset](#dataset)
- [Retrieval Models](#retrieval-models)
- [Output](#output)
- [Evaluation](#evaluation)
- [Conclusions](#Conclusions)
- [License](#license)

## Overview

The IR system is designed to handle a collection of documents and a set of user queries. It supports three retrieval models:

1. Vector Space Model (VSM): Documents are ranked based on the sum of Term Frequency-Inverse Document Frequency (TF-IDF) scores for terms present in the query.

2. BM25: Documents are ranked using the BM25 scoring function, which takes into account term frequency, document frequency, and document length.

3. Okapi BM25 Language Model: Documents are ranked using a language model approach that considers term frequency and collection frequency.

## Requirements

- Python 3.x
- NLTK (Natural Language Toolkit) library and download 'stopwords' & 'plunkt'
- PyTrec_Eval library


You can install the required dependencies using pip:

```bash
pip install nltk
pip install pytrec_eval
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/pradipm142/DCU_IR.git
cd IR-System
```

2. Make sure you have the required dataset files in the appropriate path. The dataset should include:
   - `cran.all.400`: The Cranfield collection containing the documents.
   - `cran.qry`: The file containing user queries.
   - `cranqrel`: The relevance judgments for queries and documents.

3. Update the `data_path` variable in the code to specify the path to your dataset:

```python
data_path = '/path/to/your/dataset'
```

4. Run the main script to execute the IR system and retrieve ranked documents for each query:

```bash
python simple_ir_system.py
```

## Dataset

The IR system expects the Cranfield dataset, which includes a collection of documents, user queries, and relevance judgments. The dataset can be obtained from [Cranfield Collection on Digital Library Research](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/).

## Retrieval Models

The system supports three retrieval models:
1. Vector Space Model (VSM)
2. BM25
3. Okapi BM25 Language Model

Each model calculates scores for documents based on different ranking functions.

## Output

The ranked documents for each query are stored in separate output files within the `/output` directory. The files follow the format:
```
{query_id} {iter} {doc_id} {rank} {similarity:.6f} {run_id}
```

## Evaluation

The evaluation of retrieval models can be performed using pytrec_eval, which compares the ranked documents to the ground truth relevance judgments. The evaluation Metrics are used are Mean Average Precision (MAP), Precision at 5 (P@5), and Normalized Discounted Cumulative Gain (NDCG). The evaluation results are written to separate files in the `/output` directory and printed in the command line output.

## Conclusions

In conclusion, this project has achieved its primary objective of developing a simple Information Retrieval (IR) system using python, which is a nice contribution towards helping in addressing the challenges of IR systems posed by the exponential growth of digital information. In this project, I explored and evaluated three prominent retrieval models and throughout this exploration, I learned a lot and gained valuable insights of the strengths and weaknesses of each model, contributing to my understanding of their effectiveness in document ranking and retrieval tasks.
However, while the project achieved its objectives, there are several areas for possible improvements both in the project and code implementations which can be further explored:

1. Scalability: The current implementation of the IR system is suitable for small to medium-sized document collections. For handling larger datasets, optimizations in data structures and indexing algorithms may be required to enhance the system's scalability.
2. Query Processing: The project employed basic tokenization and pre-processing techniques for query processing. Advanced natural language processing (NLP) techniques, such as part-of-speech tagging and lemmatization, could be incorporated to further improve the system's understanding of user queries and enhance retrieval accuracy.
3. Advanced Retrieval Models: While VSM, BM25, and BM25 LM are widely used and effective, the project could be extended to explore more advanced retrieval models, such as neural network-based approaches like BERT or transformer models.
4. User Interface: The current implementation provides results in a command-line interface. Adding a user-friendly graphical user interface (GUI) could enhance the overall user experience and make the system more accessible.
5. Relevance Feedback: Implementing relevance feedback mechanisms, such as Rocchio's algorithm, could allow users to provide feedback on retrieved results. This feedback can then be used to re-rank the documents and provide more personalized and relevant results.
6. Performance Optimization: Fine-tuning the code implementation for performance optimization can lead to faster document retrieval and ranking. Techniques like caching and parallel processing can be explored to reduce response times and improve system efficiency.
7. Error Handling and Validation: Strengthening error handling and input validation in the code can make the system more robust and less prone to unexpected crashes or incorrect results.
8. Cross-Validation: Conducting cross-validation on different datasets can further validate the performance of the retrieval models and provide a more comprehensive evaluation.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to fork and modify the code to suit your specific needs.

If you have any questions or feedback, please feel free to open an issue or reach out to the maintainers.

---

**Note:** Update the repository URL, username, and other details according to your repository information.

Feel free to modify the content or add any other information that you think would be helpful to users using your IR system.
