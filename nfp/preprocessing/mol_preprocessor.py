from typing import Optional, Callable, Hashable

import networkx as nx
import numpy as np
import rdkit.Chem
import tensorflow as tf

from nfp.preprocessing import features
from nfp.preprocessing.preprocessor import Preprocessor
from nfp.preprocessing.tokenizer import Tokenizer


class MolPreprocessor(Preprocessor):
    def __init__(self,
                 atom_features: Optional[Callable[[rdkit.Chem.Atom], Hashable]] = None,
                 bond_features: Optional[Callable[[rdkit.Chem.Bond], Hashable]] = None,
                 output_dtype: str = 'int32',
                 ) -> None:

        self.atom_tokenizer = Tokenizer()
        self.bond_tokenizer = Tokenizer()

        if atom_features is None:
            atom_features = features.atom_features_v1

        if bond_features is None:
            bond_features = features.bond_features_v1

        self.atom_features = atom_features
        self.bond_features = bond_features
        self.output_dtype = output_dtype

    def create_nx_graph(self, mol: rdkit.Chem.Mol, **kwargs) -> nx.DiGraph:
        g = nx.Graph(mol=mol)
        g.add_nodes_from(((atom.GetIdx(), {'atom': atom}) for atom in mol.GetAtoms()))
        g.add_edges_from((
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {'bond': bond})
            for bond in mol.GetBonds()))
        return nx.DiGraph(g)

    def get_edge_features(self, edge_data: list, max_num_edges) -> {str: np.ndarray}:
        bond_feature_matrix = np.zeros(max_num_edges, dtype=self.output_dtype)
        for n, (start_atom, end_atom, bond_dict) in enumerate(edge_data):
            bond_feature_matrix[n] = self.bond_tokenizer(self.bond_features(
                bond_dict['bond'], flipped=(start_atom == bond_dict['bond'].GetBeginAtomIdx())))
        if len(edge_data) == 0:
            bond_feature_matrix[0] = self.bond_tokenizer('self-link')
        return {'bond': bond_feature_matrix}

    def get_node_features(self, node_data: list, max_num_nodes) -> {str: np.ndarray}:
        atom_feature_matrix = np.zeros(max_num_nodes, dtype=self.output_dtype)
        for n, atom_dict in node_data:
            atom_feature_matrix[n] = self.atom_tokenizer(self.atom_features(atom_dict['atom']))
        return {'atom': atom_feature_matrix}

    def get_graph_features(self, graph_data: dict) -> {str: np.ndarray}:
        return {}

    @property
    def atom_classes(self) -> int:
        """ The number of atom types found (includes the 0 null-atom type) """
        return self.atom_tokenizer.num_classes + 1

    @property
    def bond_classes(self) -> int:
        """ The number of bond types found (includes the 0 null-bond type) """
        return self.bond_tokenizer.num_classes + 1

    @property
    def output_signature(self) -> {str: tf.TensorSpec}:
        return {'atom': tf.TensorSpec(shape=(None,), dtype=self.output_dtype),
                'bond': tf.TensorSpec(shape=(None,), dtype=self.output_dtype),
                'connectivity': tf.TensorSpec(shape=(None, 2), dtype=self.output_dtype)}

    @property
    def padding_values(self) -> {str: tf.constant}:
        """Defaults to zero for each output"""
        return {key: tf.constant(0, dtype=self.output_dtype)
                for key in self.output_signature.keys()}

    @property
    def tfrecord_features(self) -> {str: tf.io.FixedLenFeature}:
        """For loading preprocessed inputs from a tf records file"""
        return {key: tf.io.FixedLenFeature([], dtype=self.output_dtype if len(val.shape) == 0 else tf.string)
                for key, val in self.output_signature.items()}


class SmilesPreprocessor(MolPreprocessor):
    def __init__(self, *args, explicit_hs: bool = True, **kwargs):
        super(SmilesPreprocessor, self).__init__(*args, **kwargs)
        self.explicit_hs = explicit_hs

    def create_nx_graph(self, smiles: str, **kwargs) -> nx.DiGraph:
        mol = rdkit.Chem.MolFromSmiles(smiles)
        if self.explicit_hs:
            mol = rdkit.Chem.AddHs(mol)
        return super(SmilesPreprocessor, self).create_nx_graph(mol, **kwargs)


class BondIndexPreprocessor(MolPreprocessor):
    def get_edge_features(self, edge_data: list, max_num_edges) -> {str: np.ndarray}:
        bond_indices = np.zeros(max_num_edges, dtype=self.output_dtype)
        for n, (_, _, edge_dict) in enumerate(edge_data):
            bond_indices[n] = edge_dict['bond'].GetIdx()
        edge_features = super(BondIndexPreprocessor, self).get_edge_features(edge_data, max_num_edges)
        return {'bond_indices': bond_indices, **edge_features}

    @property
    def output_signature(self) -> {str: tf.TensorSpec}:
        signature = super(BondIndexPreprocessor, self).output_signature
        signature['bond_indices'] = tf.TensorSpec(shape=(None,), dtype=self.output_dtype)
        return signature


class SmilesBondIndexPreprocessor(SmilesPreprocessor, BondIndexPreprocessor):
    pass
