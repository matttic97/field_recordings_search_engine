import sys
import argparse
import io
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from pathlib import Path
from bktree import BKTree
from helper import fuzzy_ratio_distance


def compute_tfidf(corpus, stop_words):
    """
    Compute TF-IDF matrix and feature map for a given corpus.

    Parameters
    ----------
    corpus : list of str
        List of documents to analyze.
    stop_words : set
        Set of stop words to exclude from analysis.

    Returns
    -------
    tuple
        tfidf_array (ndarray): Matrix representing TF-IDF scores.
        feature_map (dict): Mapping of terms to column indices in tfidf_array.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    tfidf_array = tfidf_matrix.toarray()
    feature_names = tfidf_vectorizer.get_feature_names_out()
    feature_map = {feature_names[i]: i for i in range(len(feature_names))}
    
    return tfidf_array, feature_map

def index_files(text_path, stop_words):
    """
    Builds BKTree index, SymSpell index and computes TF-IDF based on text files.

    Parameters
    ----------
    text_path : str
        Directory containing text files to process.
    stop_words : set
        Set of stop words to ignore during processing.

    Returns
    -------
    tuple
        tfidf, feature_map, word_documents : (ndarray, dict, defaultdict)
    """
    print('Processing...')

    bktree = BKTree(fuzzy_ratio_distance)

    slovenian_alphabet = set("abcčćdđeéfghijklmnoópqrsštuvwxyzž ")
    word_documents = defaultdict(list)
    word_counts = defaultdict(int)
    documents = [[]] * 1012

    for transcription in Path(text_path).rglob('*.txt'):
        text = io.open(transcription, mode='r', encoding="utf-8").read().lower()
        words = ''.join([c for c in text if c in slovenian_alphabet]).split()
        doc = int(str(transcription.name).split('_')[1]) - 1

        entity_counts = defaultdict(int)
        for word in words:
            if word in stop_words:
                continue
            entity_counts[word] += 1           

        for word, count in sorted(entity_counts.items(), key=lambda item: item[1], reverse=True):
            if word not in word_documents:
                bktree.add(word)
            word_documents[word].append(doc)
            word_counts[word] += count

        documents[doc] = " ".join([word for word in words if word not in stop_words])

    print('Computing TF-IDF...')
    tfidf, feature_map = compute_tfidf(documents, list(stop_words.keys()))

    return bktree, tfidf, feature_map, word_documents, word_counts

def run_indexing(transcriptions_dir, index_output_dir, stop_words_path):
    if stop_words_path == "":
        stop_words = {}
    else:
        stop_words = { w : 0 for w in io.open(stop_words_path, mode='r', encoding="utf-8").read().split(',')}
    
    bktree, tfidf, feature_map, word_documents, word_counts = index_files(transcriptions_dir, stop_words)

    print("Exporting indexes...")

    bktree.save_to_file(f'{index_output_dir}/bktree.json')

    with open(f'{index_output_dir}/tfidf.pickle', 'wb') as handle:
        pickle.dump(tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{index_output_dir}/feature_map.pickle', 'wb') as handle:
        pickle.dump(feature_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{index_output_dir}/word_documents.pickle', 'wb') as handle:
        pickle.dump(word_documents, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{index_output_dir}/word_counts.txt', 'w') as file:
        for w, c in word_counts.items():
            file.write(f'{w} {c}\n')
    
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run bk-tree and SymSpell indexing on transcriptions.")
    parser.add_argument("--transcriptions_dir", type=str, required=True, help="Path to the directory containing transcription to be indexed.")
    parser.add_argument("--index_output_dir", type=str, required=True, help="Path to the directory to store indexes.")
    parser.add_argument("--stop_words_path", type=str, default="", help="Path to stop words text file.")
    
    args = parser.parse_args()

    run_indexing(args.transcriptions_dir, args.index_output_dir, args.stop_words_path)