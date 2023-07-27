# Simple Information Retrieval (IR) System with VSM, BM25, and BM25 Language Model
## An Exploration of Information Retrieval Techniques for Document Ranking and Evaluation 


This repository contains the code for a simple Information Retrieval (IR) system that retrieves relevant documents from a collection based on user queries using different retrieval models. The system is implemented in Python and includes functionalities for data loading, preprocessing, building the inverted index, ranking documents, and evaluating retrieval models. This project was done as part of my Mechanics of Search course module assignment under Dublin City University (DCU).

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Dataset](#dataset)
- [Retrieval Models](#retrieval-models)
- [Output](#output)
- [Evaluation](#evaluation)
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

The evaluation of retrieval models can be performed using trec_eval, which compares the ranked documents to the ground truth relevance judgments. The evaluation results are written to separate files in the `/output` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to fork and modify the code to suit your specific needs.

If you have any questions or feedback, please feel free to open an issue or reach out to the maintainers.

---

**Note:** Update the repository URL, username, and other details according to your repository information.

Feel free to modify the content or add any other information that you think would be helpful to users using your IR system.
