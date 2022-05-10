from typing import Callable, Dict, Hashable, Optional

import networkx as nx
import numpy as np
import rdkit.Chem

try:
    import tensorflow as tf
except ImportError:
    tf = None

from nfp.preprocessing import features
from nfp.preprocessing.preprocessor import Preprocessor
from nfp.preprocessing.tokenizer import Tokenizer

# This class handles diradical fragments described using global features - ie,
# Cartesian distance between radical centers and the length of any existing conjugation
# path between the radical centers.

class DiradPreprocessor(MolPreprocessor):
    def __init__(
        self,
        atom_features: Optional[Callable[[rdkit.Chem.Atom], Hashable]] = None,
        bond_features: Optional[Callable[[rdkit.Chem.Bond], Hashable]] = None,
        global_features: Optional[Callable[[list], Hashable]] = None,
        **kwargs,
    ) -> None:
        super(MolPreprocessor, self).__init__(**kwargs)

        self.atom_tokenizer = Tokenizer()
        self.bond_tokenizer = Tokenizer()
        self.global_tokenizer = Tokenizer()

        if atom_features is None:
            atom_features = features.atom_features_v1

        if bond_features is None:
            bond_features = features.bond_features_v1

        if global_features is None:
            global_features = features.global_features_v1

        self.atom_features = atom_features
        self.bond_features = bond_features
        self.global_features = global_features


    def create_nx_graph(self, mol: rdkit.Chem.Mol, **kwargs) -> nx.DiGraph:
        g = nx.Graph(mol=mol)
        g.add_nodes_from(((atom.GetIdx(), {"atom": atom}) for atom in mol.GetAtoms()))
        g.add_edges_from(
            (
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {"bond": bond})
                for bond in mol.GetBonds()
            )
        )
        return nx.DiGraph(g)

    def get_edge_features(
        self, edge_data: list, max_num_edges
    ) -> Dict[str, np.ndarray]:
        bond_feature_matrix = np.zeros(max_num_edges, dtype=self.output_dtype)
        for n, (start_atom, end_atom, bond_dict) in enumerate(edge_data):
            flipped = start_atom == bond_dict["bond"].GetEndAtomIdx()
            bond_feature_matrix[n] = self.bond_tokenizer(
                self.bond_features(bond_dict["bond"], flipped=flipped)
            )

        return {"bond": bond_feature_matrix}

    def get_node_features(
        self, node_data: list, max_num_nodes
    ) -> Dict[str, np.ndarray]:
        atom_feature_matrix = np.zeros(max_num_nodes, dtype=self.output_dtype)
        for n, atom_dict in node_data:
            atom_feature_matrix[n] = self.atom_tokenizer(
                self.atom_features(atom_dict["atom"])
            )
        return {"atom": atom_feature_matrix}

    def get_graph_features(self, graph_data: dict) -> Dict[str, np.ndarray]:
        return {}

    @property
    def atom_classes(self) -> int:
        """The number of atom types found (includes the 0 null-atom type)"""
        return self.atom_tokenizer.num_classes + 1

    @property
    def bond_classes(self) -> int:
        """The number of bond types found (includes the 0 null-bond type)"""
        return self.bond_tokenizer.num_classes + 1

    @property
    def glob_classes(self) -> int:
        """"Number of global feature types found"""
        return self.global_tokenizer.num_classes + 1

    @property
    def output_signature(self) -> Dict[str, tf.TensorSpec]:
        if tf is None:
            raise ImportError("Tensorflow was not found")
        return {
            "atom": tf.TensorSpec(shape=(None,), dtype=self.output_dtype),
            "bond": tf.TensorSpec(shape=(None,), dtype=self.output_dtype),
            "global": tf.TensorSpec(shape=(None,), dtype = self.output_dtype),
            "connectivity": tf.TensorSpec(shape=(None, 2), dtype=self.output_dtype),
        }

    @property
    def padding_values(self) -> Dict[str, tf.constant]:
        """Defaults to zero for each output"""
        if tf is None:
            raise ImportError("Tensorflow was not found")
        return {
            key: tf.constant(0, dtype=self.output_dtype)
            for key in self.output_signature.keys()
        }

    @property
    def tfrecord_features(self) -> Dict[str, tf.io.FixedLenFeature]:
        """For loading preprocessed inputs from a tf records file"""
        if tf is None:
            raise ImportError("Tensorflow was not found")
        return {
            key: tf.io.FixedLenFeature(
                [], dtype=self.output_dtype if len(val.shape) == 0 else tf.string
            )
            for key, val in self.output_signature.items()
        }
