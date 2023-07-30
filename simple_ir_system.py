import os
import nltk
import math
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from operator import itemgetter
import pytrec_eval
import pandas as pd

# Download NLTK "StopWords" and "Punkt" packages
nltk.download('stopwords')
nltk.download('punkt')

# Set the file paths
data_path = '/cranfieldCollection/'
collection_file = os.path.join(data_path, 'cran.all.1400')
queries_file = os.path.join(data_path, 'cran.qry')
relevance_file = os.path.join(data_path, 'cranqrel')
original_document_path = os.path.join(data_path, 'originalDocument')
tokenized_document_path = os.path.join(data_path, 'tokenizedDocument')
output_dir = os.path.join(data_path, 'output')

# Load the Cranfield collection
def load_cranfield_collection(collection_file):
    with open(collection_file, 'r', encoding='utf-8') as file:
        documents = file.read().split('.I')[1:]

    collection = {}
    for i, doc in enumerate(documents, start=1):
        '''
        #Exiting the loop after 10 iterations
        if i > 10:  # Exit the loop after 10 iterations
            print("Exiting the loop after 10 iterations.")
            break
        '''
        doc_id, content = doc.split('.T\n')
        doc_id = doc_id.strip()
        content = content.strip()
        collection[doc_id] = content

        # Create a new folder "originalDocument" if it doesn't exist
        if not os.path.exists(original_document_path):
            os.makedirs(original_document_path)

        # Write the content to a new text file with doc_id as the file name
        doc_file = os.path.join(original_document_path, f'{doc_id}.txt')
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(content)

    return collection
    
# Function to pre-process the source documents
def preprocess_document(doc_id, doc_content):
    tokens = word_tokenize(doc_content.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    # Create a new folder "originalDocument" if it doesn't exist
    if not os.path.exists(tokenized_document_path):
        os.makedirs(tokenized_document_path)
    
    # Write the tokens to a new text file with doc_id as the file name
    doc_file = os.path.join(tokenized_document_path, f'{doc_id}.txt')
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(' '.join(stemmed_tokens))

    return stemmed_tokens
    
# Load the queries
def load_queries(queries_file):
    with open(queries_file, 'r', encoding='utf-8') as file:
        queries = file.read().split('.I')[1:]

    queries_dict = {}
    for query in queries:
        query_id, content = query.split('.W\n')
        queries_dict[query_id.strip()] = content.strip()

    return queries_dict
    
# Function to calculate term frequency (TF)
def calculate_tf(query_tokens):
    tf = defaultdict(int)
    for token in query_tokens:
        tf[token] += 1
    return tf
    
# Function to calculate inverse document frequency (IDF)
def calculate_idf(inverted_index, num_documents):
    idf = {}
    for token, doc_ids in inverted_index.items():
        idf[token] = math.log(num_documents / (len(doc_ids) + 1))
    return idf
    
# Function to build inverted index
def build_inverted_index(documents):
    inverted_index = defaultdict(list)
    for doc_id, doc_content in documents.items():
        tokens = preprocess_document(doc_id, doc_content)
        for token in tokens:
            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)
    return inverted_index

# Function to build the term-document matrix for the Vector Space Model
def build_term_document_matrix(documents, inverted_index, idf):
    term_document_matrix = defaultdict(dict)
    for doc_id, doc_content in documents.items():
        tokens = preprocess_document(doc_id, doc_content)
        tf = calculate_tf(tokens)
        for token, freq in tf.items():
            tfidf = freq * idf[token]
            term_document_matrix[token][doc_id] = tfidf
    return term_document_matrix
    
# Function to calculate the BM25 score for a query and a document
def calculate_bm25_score(query_tokens, doc_tokens, inverted_index, num_documents, avg_doc_length, k1=1.5, b=0.75):
    score = 0
    doc_len = len(doc_tokens)
    for token in query_tokens:
        if token in inverted_index:
            df = len(inverted_index[token])
            idf = math.log((num_documents - df + 0.5) / (df + 0.5) + 1.0)
            tf = doc_tokens.count(token)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_length))
            score += idf * (numerator / denominator)
    return score
    
# Function to calculate the BM25 Language Model score for a query and a document
def calculate_language_model_score(query_tokens, doc_tokens, token_freq, total_token_count, lambd=0.5):
    score = 1
    doc_len = len(doc_tokens)
    for token in query_tokens:
        tf = doc_tokens.count(token)
        prob = (1 - lambd) * (tf / doc_len) + lambd * (token_freq[token] / total_token_count)
        score *= prob
    return score
    
# Function to rank documents using the Vector Space Model
def rank_documents_vsm(query_tokens, term_document_matrix):
    scores = defaultdict(float)
    for token in query_tokens:
        if token in term_document_matrix:
            for doc_id, tfidf in term_document_matrix[token].items():
                scores[doc_id] += tfidf
    ranked_docs = sorted(scores.items(), key=itemgetter(1), reverse=True)
    return ranked_docs
    
# Function to rank documents using BM25
def rank_documents_bm25(query_tokens, documents, inverted_index, num_documents, avg_doc_length):
    scores = defaultdict(float)
    for doc_id, doc_content in documents.items():
        doc_tokens = preprocess_document(doc_id, doc_content)
        score = calculate_bm25_score(query_tokens, doc_tokens, inverted_index, num_documents, avg_doc_length)
        scores[doc_id] = score
    ranked_docs = sorted(scores.items(), key=itemgetter(1), reverse=True)
    return ranked_docs
    
# Function to rank documents using Language Model (LM)
def rank_documents_language_model(query_tokens, documents, token_freq, total_token_count):
    scores = defaultdict(float)
    for doc_id, doc_content in documents.items():
        doc_tokens = preprocess_document(doc_id, doc_content)
        score = calculate_language_model_score(query_tokens, doc_tokens, token_freq, total_token_count)
        scores[doc_id] = score
    ranked_docs = sorted(scores.items(), key=itemgetter(1), reverse=True)
    return ranked_docs
    
# Function to write the output file in the required format
def write_output_file(query_id, ranked_docs, model_name, results_dir):
    # Create a new folder "results" if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_file = os.path.join(results_dir, f'{model_name}_output.txt')
    with open(output_file, 'a') as f:
        for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
            # As the values of iter, rank, and run_id are irrelevant for trec_eval, we can set them to fixed values.
            # For example, we can set iter and run_id to 0, and rank to the rank of the document in the ranked list.
            iter_value = 0
            rank_value = rank
            similarity_value = score
            run_id_value = "IR_System"
            f.write(f"{query_id} {iter_value} {doc_id} {rank_value} {similarity_value:.6f} {run_id_value}\n")
         
         
# Load the relevance judgments
def load_relevance_judgments(relevance_file):
    relevance_judgments = defaultdict(dict)
    with open(relevance_file, 'r', encoding='utf-8') as f_qrel:
        for line in f_qrel:
            parts = line.strip().split()
            if len(parts) != 3:
                # Skip lines with incorrect format
                continue
            query_id, object_id, relevance = parts
            try:
                relevance_judgments[query_id][object_id] = int(relevance)
            except ValueError:
                # Skip lines with incorrect relevance values
                continue
    return relevance_judgments
    
# Custom implementation of parse_run of pytrec_eval to handle multiple document Ids for the same query and proceed without any error
def parse_run(f_run):
    run = defaultdict(dict)
    for line in f_run:
        query_id, _, object_id, _, score, _ = line.strip().split()
        run[query_id][object_id] = float(score)
    return run
    
# Function to evaluate retrieval models using pytrec_eval
def evaluate_models(model_names, output_dir, relevance_judgments):
    evaluation_results = {}
    for model_name in model_names:
        output_file = os.path.join(output_dir, f'{model_name}_output.txt')
        with open(output_file, 'r') as f:
          # using the custom parse_run to handle duplicate document Id for the same query scenarios
          run = parse_run(f)
          #run = pytrec_eval.parse_run(f)
        
        # Using MAP, P@5, and NDCG measures for the evaluation
        evaluator = pytrec_eval.RelevanceEvaluator(relevance_judgments, {'map', 'P_5', 'ndcg'})    
        model_scores = evaluator.evaluate(run)
        #print(model_scores.values()) 
        # Store the evaluation results in a dictionary
        evaluation_results[model_name] = model_scores        
    
    return evaluation_results  


# The MAIN Function to run the IR System and Save & Print the evaluation results
if __name__ == "__main__":    
    # Load the Cranfield collection
    documents = load_cranfield_collection(collection_file)

    # Preprocess the documents and build the inverted index
    inverted_index = build_inverted_index(documents)

    # Calculate inverse document frequency (IDF)
    num_documents = len(documents)
    idf = calculate_idf(inverted_index, num_documents)

    # Load the queries
    queries = load_queries(queries_file)    

    # Build the term-document matrix for Vector Space Model
    term_document_matrix = build_term_document_matrix(documents, inverted_index, idf)

    # Rank documents using Vector Space Model
    for query_id, query_content in queries.items():
        query_tokens = preprocess_document(query_id, query_content)
        ranked_docs_vsm = rank_documents_vsm(query_tokens, term_document_matrix)
        write_output_file(query_id, ranked_docs_vsm[:100], 'vsm', output_dir)
    
    # Rank documents using BM25
    avg_doc_length = sum(len(preprocess_document(doc_id, doc_content)) for doc_id, doc_content in documents.items()) / num_documents
    for query_id, query_content in queries.items():
        query_tokens = preprocess_document(query_id, query_content)
        ranked_docs_bm25 = rank_documents_bm25(query_tokens, documents, inverted_index, num_documents, avg_doc_length)
        write_output_file(query_id, ranked_docs_bm25[:100], 'bm25', output_dir)

    # Rank documents using Okapi BM25 Language Model
    token_freq = defaultdict(int)
    total_token_count = 0
    for doc_id, doc_content in documents.items():
        doc_tokens = preprocess_document(doc_id, doc_content)
        for token in doc_tokens:
            token_freq[token] += 1
            total_token_count += 1
    for query_id, query_content in queries.items():
        query_tokens = preprocess_document(query_id, query_content)
        ranked_docs_lm = rank_documents_language_model(query_tokens, documents, token_freq, total_token_count)
        write_output_file(query_id, ranked_docs_lm[:100], 'bm25_lm', output_dir)
    
    # Load the relevance judgments from the cranqrel file
    relevance_judgments = load_relevance_judgments(relevance_file)    
    #print(relevance_judgments.values())

    # Evaluate retrieval models using pytrec_eval
    model_names = ['vsm', 'bm25', 'bm25_lm']
    evaluation_results = evaluate_models(model_names, output_dir, relevance_judgments)

    # Print the evaluation results
    print("\033[1m\033[4mEvaluation Results:\033[0m")

    # Convert the nested dictionary into a DataFrame
    data = {}
    for model, model_results in evaluation_results.items():
        for query, query_results in model_results.items():
            if all(key in query_results for key in ['map', 'P_5', 'ndcg']):
                data[(model, query)] = query_results

    df = pd.DataFrame.from_dict(data, orient='index')
    # Rename the columns and index (row) with custom names
    df.columns = ['MAP_Score', 'P@5_Score', 'NDCG_Score']
    df.index.name = 'Model'

    # Set the number of decimal places
    df = df.round(4)
    # Transpose the DataFrame to get a more readable format
    #df = df.T

    # Save the Evaluation Results to a file
    df.to_csv(output_dir + 'evaluation_results.csv')
    print(df)

    # Load the relevance judgments from the cranqrel file
    relevance_judgments = load_relevance_judgments(relevance_file)    
    #print(relevance_judgments.values())

    # Evaluate retrieval models using pytrec_eval
    model_names = ['vsm', 'bm25', 'bm25_lm']
    evaluation_results = evaluate_models(model_names, output_dir, relevance_judgments)
