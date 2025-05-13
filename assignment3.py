import os
import json
import re
import time
import nltk
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
import pickle
from nltk.stem import PorterStemmer
import math

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class InvertedIndex:
    def __init__(self, corpus_dir):
        """
        Initialize the inverted index builder
        
        Args:
            corpus_dir (str): Path to directory containing the corpus files
        """
        self.corpus_dir = corpus_dir
        self.stemmer = PorterStemmer()
        
        # Main inverted index: {token: [(doc_id, tf, positions, importance_score)]}
        self.index = defaultdict(list)
        
        # Document mapping: {doc_id: {"url": url, "path": file_path}}
        self.doc_map = {}
        
        # Document count
        self.doc_count = 0
        
        # Total number of tokens in all documents
        self.total_tokens = 0
        
        # Document frequency: {token: count of documents containing this token}
        self.doc_freq = defaultdict(int)

    def tokenize(self, text):
        """
        Tokenize text into alphanumeric sequences
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            list: List of tokens
        """
        # Tokenize alphanumeric sequences
        tokens = re.findall(r'\w+', text.lower())
        # Stem tokens
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return stemmed_tokens
    
    def extract_content_with_importance(self, soup):
        """
        Extract content from HTML with importance markers without duplication
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            
        Returns:
            dict: Dictionary with text content and importance markers
        """
        # Initialize result structure
        result = {
            "text": "",
            "important_tokens": set()
        }
        
        # First identify important tokens by looking at specific tags
        important_elements = []
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            important_elements.append(title_tag)
        
        # Extract headings
        for heading_tag in soup.find_all(['h1', 'h2', 'h3']):
            if heading_tag.string:
                important_elements.append(heading_tag)
        
        # Extract bold text
        for bold_tag in soup.find_all(['strong', 'b']):
            if bold_tag.string:
                important_elements.append(bold_tag)
        
        # Process all important elements and extract their tokens
        for element in important_elements:
            text = element.get_text(strip=True)
            tokens = self.tokenize(text)
            result["important_tokens"].update(tokens)
        
        # Extract all text content once (no duplication)
        body_tag = soup.find('body')
        if body_tag:
            result["text"] = body_tag.get_text(separator=' ', strip=True)
        else:
            # If no body tag, extract all text from the document
            result["text"] = soup.get_text(separator=' ', strip=True)
        
        return result
    
    def process_document(self, doc_id, file_path):
        """
        Process a document and add its tokens to the inverted index
        
        Args:
            doc_id (int): Document ID
            file_path (str): Path to the document file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            url = data.get('url', '')
            content = data.get('content', '')
            
            # Store document mapping
            self.doc_map[doc_id] = {
                "url": url,
                "path": file_path
            }
            
            # Parse HTML content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract content with importance markers
            extracted_content = self.extract_content_with_importance(soup)
            text = extracted_content["text"]
            important_tokens = extracted_content["important_tokens"]
            
            # Tokenize text
            tokens = self.tokenize(text)
            
            # Skip empty documents
            if not tokens:
                return
            
            # Count token frequencies in document
            token_freq = Counter(tokens)
            
            # Track unique tokens in this document for document frequency calculation
            unique_tokens = set(tokens)
            
            # Update document frequency for each unique token
            for token in unique_tokens:
                self.doc_freq[token] += 1
            
            # Process each token
            for token, freq in token_freq.items():
                # Calculate positions of the token in the document
                positions = [i for i, t in enumerate(tokens) if t == token]
                
                # Calculate importance score (1 if token is important, 0 otherwise)
                importance_score = 1 if token in important_tokens else 0
                
                # Add posting to inverted index
                self.index[token].append((doc_id, freq, positions, importance_score))
            
            # Update total token count
            self.total_tokens += len(tokens)
            
            # Update document count
            self.doc_count += 1
            
            # Print progress every 1000 documents
            if self.doc_count % 1000 == 0:
                print(f"Processed {self.doc_count} documents...")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def build_index(self):
        """
        Build the inverted index for the corpus
        """
        start_time = time.time()
        
        print(f"Building inverted index for corpus in {self.corpus_dir}...")
        
        # Walk through the corpus directory
        for domain_dir in os.listdir(self.corpus_dir):
            domain_path = os.path.join(self.corpus_dir, domain_dir)
            
            # Skip if not a directory
            if not os.path.isdir(domain_path):
                continue
            
            print(f"Processing domain: {domain_dir}")
            
            # Process each file in the domain directory
            for file_name in os.listdir(domain_path):
                if not file_name.endswith('.json'):
                    continue
                    
                file_path = os.path.join(domain_path, file_name)
                
                # Generate document ID
                doc_id = self.doc_count
                
                # Process the document
                self.process_document(doc_id, file_path)
        
        end_time = time.time()
        print(f"Index built in {end_time - start_time:.2f} seconds")
        print(f"Indexed {self.doc_count} documents")
        print(f"Found {len(self.index)} unique tokens")
        
    def save_index(self, index_file="inverted_index.pkl"):
        """
        Save the inverted index to a file
        
        Args:
            index_file (str): Path to save the index
        """

        #Pickle starts here btw. Use pickle to simulate what it is like in disk memory

        index_data = {
            "index": dict(self.index),
            "doc_map": self.doc_map,
            "doc_count": self.doc_count,
            "total_tokens": self.total_tokens,
            "doc_freq": dict(self.doc_freq)
        }
        
        with open(index_file, 'wb') as f:
            pickle.dump(index_data, f)
            
        # Get file size
        index_size = os.path.getsize(index_file) / 1024  # Size in KB
        print(f"Index saved to {index_file}")
        print(f"Index size: {index_size:.2f} KB")
        
        
        #Pickle Ends Here

        # Save analytics to a report file
        with open("index_report.txt", 'w') as f:
            f.write(f"Number of indexed documents: {self.doc_count}\n")
            f.write(f"Number of unique tokens: {len(self.index)}\n")
            f.write(f"Total size of index on disk: {index_size:.2f} KB\n")
            
            # Add some additional analytics
            f.write(f"\nAdditional Analytics:\n")
            f.write(f"Total number of tokens: {self.total_tokens}\n")
            f.write(f"Average tokens per document: {self.total_tokens / self.doc_count if self.doc_count > 0 else 0:.2f}\n")
            
            # Top 20 most frequent tokens
            f.write(f"\nTop 20 most frequent tokens:\n")
            token_frequencies = [(token, sum(post[1] for post in postings)) for token, postings in self.index.items()]
            token_frequencies.sort(key=lambda x: x[1], reverse=True)
            for i, (token, freq) in enumerate(token_frequencies[:20], 1):
                f.write(f"{i}. {token}: {freq}\n")
            
            # Top 20 tokens that appear in most documents
            f.write(f"\nTop 20 tokens that appear in most documents:\n")
            doc_freq_list = [(token, freq) for token, freq in self.doc_freq.items()]
            doc_freq_list.sort(key=lambda x: x[1], reverse=True)
            for i, (token, freq) in enumerate(doc_freq_list[:20], 1):
                f.write(f"{i}. {token}: {freq}\n")

def main():
    # Set the corpus directory
    corpus_dir = "DEV"  # Change this to the path of your corpus
    
    # Create and build the inverted index
    indexer = InvertedIndex(corpus_dir)
    indexer.build_index()
    indexer.save_index()

if __name__ == "__main__":
    main()