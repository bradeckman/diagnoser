import itertools
import math


END = 'end'
YES, NO = 'yes', 'no'
ROOT_INDEX, SCORE_INDEX = 0, 1


class Node:
    """
    The Node class represents an object that contains data, as well as
    two 'children' nodes - essentially pointers to two other Node instances.
    The terms 'positive' and 'negative' are attributed to the context of
    decision trees - 'positive' meaning yes, 'negative' meaning no. With Node
    instances, we can build these trees and traverse through them by accessing
    the children nodes.
    """
    CHILDREN_KINDS = POSITIVE, NEGATIVE = 0, 1

    def __init__(self, data, positive_child=None, negative_child=None):
        """ Constructor for Node. """
        self.data = data
        self.positive_child = positive_child
        self.negative_child = negative_child

    def is_leaf(self):
        """
        Determines whether a Node instance is a leaf.
        :return: True if Node is leaf, else False
        """
        if self.positive_child is None and self.negative_child is None:
            return True

        return False

    def get_data(self):
        """ Getter for data. """
        return self.data

    def get_positive_child(self):
        """ Getter for positive child. """
        return self.positive_child

    def get_negative_child(self):
        """ Getter for negative child. """
        return self.negative_child

    def get_children(self):
        """
        Gets both children with their corresponding boolean values.
        :return: Dict in format {child_node, bool_val}
        """
        children = {self.positive_child: True, self.negative_child: False}
        return children

    def get_child(self, kind):
        """
        Getter for the specified child.
        :param kind: The kind of child (POSITIVE or NEGATIVE)
        :return: The child node
        """
        if kind == self.POSITIVE:
            return self.positive_child
        if kind == self.NEGATIVE:
            return self.negative_child

    def set_data(self, data):
        """ Setter for positive child. """
        self.data = data

    def set_positive_child(self, node):
        """ Setter for positive child. """
        self.positive_child = node

    def set_negative_child(self, node):
        """ Setter for negative child. """
        self.negative_child = node

    def set_child(self, kind, node):
        """
        Setter for the specified child.
        :param kind: The kind of child (POSITIVE or NEGATIVE)
        :param node: An object of type Node
        :return: None
        """
        if kind == self.POSITIVE:
            self.positive_child = node
        if kind == self.NEGATIVE:
            self.negative_child = node


class Record:
    """
    The Record class represents an object that contains an illness, and a list
    of the symptoms that have been associated with that illness.
    """
    def __init__(self, illness, symptoms):
        """ Constructor for Record. """
        self.illness = illness
        self.symptoms = symptoms

    def get_illness(self):
        """ Getter for illness. """
        return self.illness

    def get_symptoms(self):
        """ Getter for symptoms. """
        return self.symptoms


def parse_data(filepath):
    """
    Reads and formats data from file into Record objects
    :param filepath: A file path
    :return: A list of objects of type Record
    """
    with open(filepath) as data_file:
        records = []
        for line in data_file:
            words = line.strip().split()
            records.append(Record(words[0], words[1:]))
        return records


class Diagnoser:
    """
    The Diagnoser class represents an object that only contains a tree
    (accessed through the root), of Node instances. The class contains several
    functions that allow us to traverse through the tree's symptoms
    """
    def __init__(self, root):
        """ Constructor for Diagnoser. """
        self.root = root

    def diagnose_helper(self, symptoms, current):
        """
        Helper function for self.diagnose.
        :param symptoms: A list of symptoms
        :param current: The current node in the tree
        :return: The diagnosis
        """
        # Base case - if the current node is a leaf, return the illness.
        if current.is_leaf():
            return current.get_data()

        # If the current node's data (the symptom) is in our observed symptoms,
        # traverse to positive child; else, to negative child
        if current.get_data() in symptoms:
            return self.diagnose_helper(symptoms, current.get_positive_child())
        else:
            return self.diagnose_helper(symptoms, current.get_negative_child())

    def diagnose(self, symptoms):
        """
        Diagnoses an illness given observed symptoms.
        :param symptoms: A list of symptoms
        :return: The diagnosis
        """
        # Kick off recursive function with root node
        current = self.root

        return self.diagnose_helper(symptoms, current)

    def calculate_success_rate(self, records):
        """
        Calculates a score for the ability of our tree to diagnose accurately,
        given existing records.
        :param records: A list of Record objects
        :return: The score for this tree
        """
        correct_diagnosis = 0

        # Iterate through records and compare our diagnosis with actual
        for record in records:
            diagnosis = self.diagnose(record.get_symptoms())
            if diagnosis == record.get_illness():
                correct_diagnosis += 1

        # Calculation for score
        diagnosis_percentage = correct_diagnosis / len(records)

        return diagnosis_percentage

    def add_to_illnesses(self, illness, illnesses):
        """
        Updates the illnesses dictionary.
        :param illness: An illness
        :param illnesses: A dictionary to tally illnesses, {illness: count}
        :return: None
        """
        if illness not in illnesses:
            illnesses[illness] = 1
        else:
            illnesses[illness] += 1

    def all_illnesses_helper(self, current, illnesses):
        """
        Helper function for self.all_illnesses; Updates illnesses dictionary.
        :param current: The current node
        :param illnesses: A dictionary to tally illnesses, {illness: count}
        :return: None
        """
        # Base case - if the current node is a leaf, it is an illness
        if current.is_leaf():
            illness = current.get_data()
            self.add_to_illnesses(illness, illnesses)
            return

        # Traverse through tree while we haven't reached a leaf
        for child in current.get_children():
            self.all_illnesses_helper(child, illnesses)

    def all_illnesses(self):
        """
        Returns all illnesses within the tree.
        :return: A list of illnesses
        """
        # Kick off recursive function with root node and empty dict
        current = self.root
        illnesses = {}
        # Populate illnesses dict
        self.all_illnesses_helper(current, illnesses)

        # Format dict to list [(illness, count), sort by count, erase counts
        illnesses = list(illnesses.items())
        illnesses.sort(key=lambda a: a[1], reverse=True)
        illnesses = [illness for illness, count in illnesses]

        return illnesses

    def paths_to_illness_helper(self, illness, current, path, paths):
        """
        Helper function for self.paths_to_illness.
        :param illness: An illness
        :param current: The current node
        :param path: The current path as a list of boolean values (True if
                     positive node, else False)
        :param paths: All paths to illness until now
        :return: All paths to illness until now
        """
        # Base case - if current node is leaf and is illness, append current
        # path to possible paths
        if current.is_leaf():
            if current.get_data() == illness:
                paths.append(path)
            return

        # Create 2 new paths, one for each child, and recurse
        for child, notation in current.get_children().items():
            new_path = path[:]
            new_path.append(notation)
            self.paths_to_illness_helper(illness, child, new_path, paths)

        return paths

    def paths_to_illness(self, illness):
        """
        Returns all paths in tree to a given illness.
        :param illness: An illness
        :return: All paths of the tree that lead to the specified illness
        """
        # Initialize variables
        current = self.root
        path = []
        paths = []
        # Kick off recursive function with the root node
        self.paths_to_illness_helper(illness, current, path, paths)

        return paths


def matching_symptoms(record_symptoms, path):
    """
    Determines whether a record is consistent with a given path.
    :param record_symptoms: A list of symptoms from a Record object
    :param path: The path we've traversed until now; A list of tuples in format
                 [(symptom (node.data), bool_value)]
    :return: True if record matches, else False
    """
    for tree_symptom, is_symptomatic in path:
        # Records that have a symptom that path does not are inconsistent
        if is_symptomatic and tree_symptom not in record_symptoms:
            return False
        # Records that do not have a symptom that path does are inconsistent
        if not is_symptomatic and tree_symptom in record_symptoms:
            return False

    return True


def add(item, dict):
    """
    Adds point to tallying dictionary.
    :param item: The key in the dictionary
    :param dict: The dictionary
    :return: None
    """
    if item in dict:
        dict[item] += 1
    else:
        dict[item] = 1


def match_illness_to_symptoms(path, records):
    """
    Calculates the most probable illness for symptoms (inputed as path) given
    past records.
    :param path: The path we've traversed until now; A list of tuples in format
                 [(symptom (node.data), bool_value)]
    :param records: A list of Record objects
    :return: An illness
    """
    # Iterate over records; if symptoms are consistent with those of path,
    # record a point for that record's illness
    probable_diagnoses = {}
    for record in records:
        if matching_symptoms(record.symptoms, path):
            add(record.illness, probable_diagnoses)
    # If no records match, return None
    if not probable_diagnoses:
        return

    # The most probable illness is that which has the highest number of points
    most_probable_illness, count = max(probable_diagnoses.items(),
                                       key=lambda a: a[1])

    return most_probable_illness


def build_tree_helper(node, symptoms, path, records):
    """
    Helper function for self.build_tree.
    :param node: The current node
    :param symptoms: A list fof symptoms
    :param path: The path we've traversed until now; A list of tuples in format
                 [(symptom (node.data), bool_value)]
    :param records: A list of Record objects
    :return: None
    """
    # At each node, do the following for both positive and negative children
    for kind in Node.CHILDREN_KINDS:
        # Append to path the appropriate route taken
        new_path = path[:]
        if kind == Node.POSITIVE:
            new_path.append((node.get_data(), True))
        elif kind == Node. NEGATIVE:
            new_path.append((node.get_data(), False))

        # Base case - if we've used all symptoms, we must set leaves (the
        # appropriate illness)
        if len(symptoms) == 0:
            # Get appropriate illness, set as leaf
            illness = match_illness_to_symptoms(new_path, records)
            node.set_child(kind, Node(illness))

        else:
            # Set next symptom as child node and recurse
            node.set_child(kind, Node(symptoms[0]))
            build_tree_helper(node.get_child(kind), symptoms[1:], new_path,
                              records)


def build_tree(records, symptoms):
    """
    Builds a tree given a list of records and symptoms.
    :param records: A list of Record objects
    :param symptoms: A list of symptoms
    :return: The root of the tree (of type Node)
    """
    # Initialize path
    path = []

    if not symptoms:
        return Node(match_illness_to_symptoms(path, records))

    root = Node(symptoms[0])  # root node for our tree
    # Build tree of off root node
    build_tree_helper(root, symptoms[1:], path, records)

    return root


def optimal_tree(records, symptoms, depth):
    """
    Returns the most accurate tree given specified records and symptoms.
    :param records: A list of Record objects
    :param symptoms: A list of symptoms
    :param depth: The depth of the tree to build
    :return: The root of the tree (of type Node)
    """
    best_tree = (None, -math.inf)
    # Iterate through all combinations of symptoms of size depth
    for symptoms_combo in itertools.combinations(symptoms, depth):
        # Build a tree, diagnose and calculate score
        root = build_tree(records, symptoms_combo)
        diagnoser = Diagnoser(root)
        tree_score = diagnoser.calculate_success_rate(records)

        # Update best_tree if current best_tree is gets outscored
        if tree_score > best_tree[SCORE_INDEX]:
            best_tree = (root, tree_score)

    return best_tree[ROOT_INDEX]
