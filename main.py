import os
import pickle
import jsonlines
import hnswlib
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import time

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DATASET_PATH = 'completions.jsonl'
INDEX_FILE = 'hnsw_index.bin'
EMBEDDING_DIMENSION = 384
MAX_EF_CONSTRUCTION = 200
M_VALUE = 16
TOP_K_SIMILAR_QUESTIONS = 5

class SimilaritySearch:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.data = self.load_dataset(DATASET_PATH)
        self.num_elements = len(self.data)
        self.index = self.load_or_create_hnsw_index()

    def load_dataset(self, dataset_path):
        with jsonlines.open(dataset_path) as reader:
            return list(reader)

    def load_or_create_hnsw_index(self):
        if os.path.exists(INDEX_FILE):
            print("Loading precomputed index...")
            with open(INDEX_FILE, 'rb') as index_f:
                return pickle.load(index_f)
        else:
            print("Computing the index...")
            index = hnswlib.Index(space='cosine', dim=EMBEDDING_DIMENSION)
            self.index_embeddings(index)
            self.save_hnsw_index(index)
            return index

    def index_embeddings(self, index):
        embeddings = []
        count = 0
        for entry in tqdm(self.data, desc="Indexing", unit="data"):
            try:
                text = entry['prompt']
                embedding = self.model.encode(text, convert_to_tensor=False).tolist()
                embeddings.append(embedding)
                count += 1
            except Exception as e:
                print(e)
                print(entry)
                print(count)
                continue
        embeddings = np.array(embeddings)
        index.init_index(max_elements=self.num_elements, ef_construction=MAX_EF_CONSTRUCTION, M=M_VALUE)
        index.add_items(embeddings)

    def save_hnsw_index(self, index):
        with open(INDEX_FILE, 'wb') as index_f:
            pickle.dump(index, index_f)

    def find_similar_questions(self, query_text, top_k=TOP_K_SIMILAR_QUESTIONS):
        query_embedding = self.model.encode(query_text, convert_to_tensor=False).tolist()
        labels, distances = self.index.knn_query(query_embedding, k=top_k)
        similar_questions = [[self.data[label]['prompt'], self.data[label]['completion']] for label in labels[0]]
        return similar_questions

def main():
    similarity_search = SimilaritySearch()

    while True:
        print("Query:")
        query = input()
        if not query:
            break

        start_time = time.time()
        similar_questions = similarity_search.find_similar_questions(query)
        end_time = time.time()

        print("Similar questions:", similar_questions)
        print("Time taken:", end_time - start_time, "seconds")

if __name__ == "__main__":
    main()
