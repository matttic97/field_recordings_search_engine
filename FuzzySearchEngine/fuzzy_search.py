import io
from collections import defaultdict
from bktree import BKTree
from helper import fuzzy_ratio_distance
from symspellpy.symspellpy import SymSpell, Verbosity
import pickle
import numpy as np
import concurrent.futures
import heapq
from operator import itemgetter


_getitem1_0 = itemgetter(1, 0)
_npUnique = np.unique


class FuzzySearch:
    """
    FuzzySearch class for fuzzy text search in a large corpus.

    Uses a BK-tree and SymSpell for handling spelling variations and fuzzy matching.

    Attributes
    ----------
    stop_words : dict
        Set of stop words to exclude from search processing.
    tfidf : ndarray
        TF-IDF matrix for document feature weighting.
    feature_map : dict
        Mapping from terms to indices in the TF-IDF matrix.
    word_documents : dict
        Mapping from words to lists of document IDs containing those words.
    bktree : BKTree
        BK-tree for fast fuzzy matching of terms.
    sym_spell : SymSpell
        SymSpell instance for fast fuzzy text lookup.
    
    Methods
    -------
    score_match_query_ratio_with_penalty
        Adjusts document ranking with match score and repetition penalty.
    score_match_query_ratio
        Adjusts document ranking based on similarity scores.
    score_match_query_tfidf
        Adjusts document ranking based on TF-IDF scores.
    score_match_query_tfidf_with_penalty
        Adjusts document ranking with TF-IDF and repetition penalty.
    search_bktree
        Searches documents using the BK-tree.
    search_symspell
        Searches documents using SymSpell.
    find_relevant_documents
        Returns relevant documents for a given query.
    """

    @classmethod
    def score_match_query_ratio_with_penalty(cls, document_scores, doc_id, match_score, L, prev_docs, tfidf):
        """
        Adjusts document ranking based on match score, with repetition penalty.

        Parameters
        ----------
        document_scores : defaultdict
            Dictionary of document scores for ranking.
        doc_id : str
            Document ID to score.
        match_score : float
            Fuzzy match score for the document.
        L : float
            Length factor for query normalization.
        prev_docs : defaultdict
            Track previous document matches to apply penalty.
        tfidf : float
            TF-IDF score for the matched word in the document.
        """
        prev_docs[doc_id] += 1
        document_scores[doc_id] += match_score / (L * prev_docs[doc_id])

    @classmethod
    def score_match_query_ratio(cls, document_scores, doc_id, match_score, L, prev_docs, tfidf):
        """
        Adjusts document ranking based on fuzzy match scores.

        Parameters
        ----------
        document_scores : defaultdict
            Dictionary of document scores for ranking.
        doc_id : str
            Document ID to score.
        match_score : float
            Fuzzy match score for the document.
        L : float
            Length factor for query normalization.
        prev_docs : defaultdict
            Track previous document matches (not used here).
        tfidf : float
            TF-IDF score for the matched word in the document.
        """
        document_scores[doc_id] += match_score / L

    @classmethod
    def score_match_query_tfidf(cls, document_scores, doc_id, match_score, L, prev_docs, tfidf):
        """
        Adjusts document ranking based on TF-IDF scores.

        Parameters
        ----------
        document_scores : defaultdict
            Dictionary of document scores for ranking.
        doc_id : str
            Document ID to score.
        match_score : float
            Fuzzy match score for the document.
        L : float
            Length factor for query normalization.
        prev_docs : defaultdict
            Track previous document matches (not used here).
        tfidf : float
            TF-IDF score for the matched word in the document.
        """
        document_scores[doc_id] += match_score * tfidf / L

    @classmethod
    def score_match_query_tfidf_with_penalty(cls, document_scores, doc_id, match_score, L, prev_docs, tfidf):
        """
        Adjusts document ranking with TF-IDF and repetition penalty.

        Parameters
        ----------
        document_scores : defaultdict
            Dictionary of document scores for ranking.
        doc_id : str
            Document ID to score.
        match_score : float
            Fuzzy match score for the document.
        L : float
            Length factor for query normalization.
        prev_docs : defaultdict
            Track previous document matches to apply penalty.
        tfidf : float
            TF-IDF score for the matched word in the document.
        """
        prev_docs[doc_id] += 1
        document_scores[doc_id] += match_score * tfidf / (L * prev_docs[doc_id])

    def __init__(self, index_dir='./indexes', stop_words_path="./indexes/stop-words.txt"):
        """
        Initialize FuzzySearch with data from index files.

        Parameters
        ----------
        index_dir : str, optional
            Directory containing index files (default './indexes').
        stop_words_path : str, optional
            Path to file containing stop words, comma-separated (default "./indexes/stop-words.txt").
        """
        # Load stop words
        self.stop_words = (
            {w: 0 for w in io.open(stop_words_path, mode='r', encoding="utf-8").read().split(',')}
            if stop_words_path
            else {}
        )

        # Load serialized index data
        # with open(f'{index_dir}/tfidf.pickle', 'rb') as handle:
        #     self.tfidf = pickle.load(handle)
        # with open(f'{index_dir}/feature_map.pickle', 'rb') as handle:
        #     self.feature_map = pickle.load(handle)
        with open(f'{index_dir}/word_documents.pickle', 'rb') as handle:
            self.word_documents = pickle.load(handle)

        # Load BK-tree and SymSpell models
        self.bktree = BKTree(fuzzy_ratio_distance)
        self.bktree.load_from_file(f'{index_dir}/bktree.json')
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=5)
        self.sym_spell.load_dictionary(f'{index_dir}/word_counts.txt', term_index=0, count_index=1)

    def search_bktree(self, words, k, score_func, n):
        """
        Searches for documents using the BK-tree for fuzzy matching.

        Parameters
        ----------
        words : list
            List of query terms.
        k : int
            Number of top documents to return.
        score_func : function
            Scoring function to rank documents.
        n : int
            Limit on the number of BK-tree matches (default -1 for unlimited).

        Returns
        -------
        list
            Top-k documents ranked by similarity with corresponding scores.
        """
        L = len(words)
        document_scores = defaultdict(float)
        tolerance = 20  # Max allowable distance for fuzzy matching

        # Perform BK-tree search in parallel for each word
        _find = self.bktree.find
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_query = {executor.submit(_find, word, tolerance, n): word for word in words}
            results = [future.result() for future in concurrent.futures.as_completed(future_to_query)]

        # Rank documents based on matches
        for matches in results:
            prev_docs = defaultdict(int)
            for distance, match in matches:
                match_score = 1 / (distance + 1)
                if match not in self.word_documents: #self.feature_map:
                    continue
                for doc_id in self.word_documents[match]:
                    score_func(document_scores, doc_id, match_score, L * len(matches), prev_docs, None) #, self.tfidf[doc_id][self.feature_map[match]])

        return heapq.nlargest(k, document_scores.items(), key=_getitem1_0)

    def search_symspell(self, words, k, score_func, verbosity=Verbosity.CLOSEST):
        """
        Searches for documents using SymSpell for approximate matching.

        Parameters
        ----------
        words : list
            List of query terms.
        k : int
            Number of top documents to return.
        score_func : function
            Scoring function to rank documents.
        verbosity : Verbosity, optional
            SymSpell verbosity mode for match ranking (default Verbosity.CLOSEST).

        Returns
        -------
        list
            Top-k documents ranked by similarity with corresponding scores.
        """
        L = len(words)
        document_scores = defaultdict(float)

        # Perform SymSpell lookup for each word
        _lookup = self.sym_spell.lookup
        results = [
            [(s.distance, s.term) for s in _lookup(w, verbosity, max_edit_distance=2)]
            for w in words
        ]

        # Rank documents based on matches
        for matches in results:
            prev_docs = defaultdict(int)
            for distance, match in matches:
                match_score = 1 / (distance + 1)
                for doc_id in self.word_documents[match]:
                    score_func(document_scores, doc_id, match_score, L * len(matches), prev_docs, None)

        return heapq.nlargest(k, document_scores.items(), key=_getitem1_0)

    def find_relevant_documents(self, query, k=20, score_func=None, n=-1):
        """
        Finds relevant documents based on a query.

        Parameters
        ----------
        query : str
            Text query to search.
        k : int, optional
            Number of top documents to return (default 20).
        score_func : function,
        optional
            Scoring function to use for ranking (default `score_match_query_ratio_with_penalty`).
        n : int, optional
            Limit on the number of BK-tree matches (default -1 for unlimited).

        Returns
        -------
        list
            Top-k documents ranked by relevance.
        """

        if score_func == None:
            score_func = self.score_match_query_ratio_with_penalty

        # Preprocess query: remove stop words and get unique words
        words = _npUnique([word for word in query.split() if word not in self.stop_words])

        # For large queries, use SymSpell; otherwise, use BK-tree
        if len(words) > 86:
            return self.search_symspell(words, k, score_func)
        
        return self.search_bktree(words, k, score_func, n)
