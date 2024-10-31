# Attribution: Code adapted from https://github.com/benhoyt/pybktree under MIT license.
# Tree structure was modified, so it does not allow duplicates. Also, functions for automatic text indexing, exporting, and importing tree structures were added.
# Original source: https://github.com/benhoyt/pybktree.

from collections import deque
import json
import sys
from operator import itemgetter


sys.setrecursionlimit(100000)
_getitem0 = itemgetter(0)


class BKTree:
    """
    A class for the Burkhard-Keller tree structure.

    Attributes
    ----------
    distance_func : function
        Function to compute the distance between items.
    tree : tuple
        Root node of the BK-tree.

    Methods
    -------
    add(item)
        Adds an item to the tree structure if it is not a duplicate.
    find(item, tolerance, k)
        Finds items within a specified distance from a target item.
    to_dict(node)
        Converts the BK-tree to a dictionary format for saving.
    from_dict(data)
        Loads a tree structure from a dictionary format.
    save_to_file(filename)
        Saves the tree to a file in JSON format.
    load_from_file(filename)
        Loads the tree from a file in JSON format.
    fit_to_text(text_path, stop_words)
        Builds the BK-tree and computes TF-IDF based on text files.
    """
    
    def __init__(self, distance_func):
        """
        Initializes the BKTree with a specified distance function.

        Parameters
        ----------
        distance_func : function, optional
            Function to calculate the distance between strings.
        """
        self.distance_func = distance_func
        self.tree = None
        
    def add(self, item):
        """
        Adds an item to the tree, if it is not already present.

        Parameters
        ----------
        item : str
            The item to be added to the tree.
        """
        node = self.tree
        if node is None:
            self.tree = (item, {})
            return

        _distance_func = self.distance_func

        while True:
            parent, children = node
            distance = _distance_func(item, parent)

            # item already exists
            if distance == 0:
                break

            node = children.get(distance)
            if node is None:
                children[distance] = (item, {})
                break
    
    def find(self, item, tolerance, k=-1):
        """
        Finds items within a given distance from the input item.

        Parameters
        ----------
        item : str
            The target item to search for.
        tolerance : int
            The maximum allowable distance for search matches.
        k : int, optional
            The number of results to return (default is -1 for unlimited).

        Returns
        -------
        list
            List of tuples with distance and item.
        """
        if self.tree is None:
            return []

        found = [(0, item)]
        if len(item) <= 3:
            return found
        
        candidates = deque([self.tree])
        _candidates_popleft = candidates.popleft
        _candidates_extend = candidates.extend
        _found_append = found.append
        _distance_func = self.distance_func

        count = 0
        while candidates:
            candidate, children = _candidates_popleft()
            distance = _distance_func(candidate, item)
            if distance <= tolerance:
                _found_append((distance, candidate))
                count += 1
            
            if count == k:
                break

            if children:
                lower = distance - tolerance
                upper = distance + tolerance
                _candidates_extend(c for d, c in children.items() if lower <= d <= upper)

        found.sort(key=_getitem0)
        return found
    
    def to_dict(self, node=None):
        """
        Converts the BKTree structure to a dictionary format.

        Parameters
        ----------
        node : tuple, optional
            The current node to process (default is root node).

        Returns
        -------
        dict or None
            Dictionary representation of the BKTree or None if empty.
        """
        if node is None:
            node = self.tree
        
        if node is None:
            return None
        
        item, children = node
        return {
            'item': item,
            'children': {distance: self.to_dict(child) for distance, child in children.items()}
        }
    
    def from_dict(self, data):
        """
        Loads a BKTree structure from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary format of BKTree structure.

        Returns
        -------
        tuple
            The root node of the BKTree.
        """
        if data is None:
            return None
        
        item = data['item']
        children = {int(distance): self.from_dict(child) for distance, child in data['children'].items()}
        return (item, children)
    
    def save_to_file(self, filename):
        """
        Saves the BKTree to a file in JSON format.

        Parameters
        ----------
        filename : str
            Path to save the JSON file.
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f)
    
    def load_from_file(self, filename):
        """
        Loads the BKTree from a JSON file.

        Parameters
        ----------
        filename : str
            Path of the JSON file to load.
        """
        with open(filename, 'r') as f:
            self.tree = self.from_dict(json.load(f))
