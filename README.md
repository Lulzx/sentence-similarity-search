# Sentence Similarity Search

A Python project for performing similarity search on sentences using the SentenceTransformer library and HNSW indexing.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

This project demonstrates how to build a sentence similarity search system using the SentenceTransformer library, HNSW indexing, and a pre-trained transformer model. It allows users to find similar sentences or questions from a dataset based on a query input.

Key features of this project include:

- Sentence embedding using the SentenceTransformer model.
- Efficient similarity search using HNSW indexing.
- Loading and saving of the index to improve performance.

## Requirements

To run this project, you need the following:

- Python 3.9+
- Required Python packages (install via `pip install -r requirements.txt`):
  - sentence-transformers
  - hnswlib
  - tqdm
  - jsonlines

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/sentence-similarity-search.git
   ```

2. Navigate to the project directory:

   ```bash
   cd sentence-similarity-search
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the SentenceTransformer model:

   - You can change the model in the code by modifying the `MODEL_NAME` constant.

2. Prepare your dataset in JSONL format. Each entry should have a 'prompt' and 'completion' field.

3. Run the main script:

   ```bash
   python main.py
   ```

4. Enter a query, and the system will return the top similar questions from your dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
