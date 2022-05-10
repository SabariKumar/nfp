# The rest of the methods in this module are specific functions for computing
# atom and bond features. New ones can be easily added though, and these are
# passed directly to the Preprocessor class.


def get_ring_size(obj, max_size=12):
    if not obj.IsInRing():
        return 0
    else:
        for i in range(max_size):
            if obj.IsInRingSize(i):
                return i
        else:
            return "max"


def atom_features_v1(atom):
    """Return an integer hash representing the atom type"""

    return str(
        (
            atom.GetSymbol(),
            atom.GetDegree(),
            atom.GetTotalNumHs(),
            atom.GetImplicitValence(),
            atom.GetIsAromatic(),
        )
    )

def global_features_v1(cart_dist, conj_length):
    """Return a string typed hash of global features"""

    return str(
        (cart_dist, conj_length)
    )


def atom_features_v2(atom):
    props = [
        "GetChiralTag",
        "GetDegree",
        "GetExplicitValence",
        "GetFormalCharge",
        "GetHybridization",
        "GetImplicitValence",
        "GetIsAromatic",
        "GetNoImplicit",
        "GetNumExplicitHs",
        "GetNumImplicitHs",
        "GetNumRadicalElectrons",
        "GetSymbol",
        "GetTotalDegree",
        "GetTotalNumHs",
        "GetTotalValence",
    ]

    atom_type = [getattr(atom, prop)() for prop in props]
    atom_type += [get_ring_size(atom)]

    return str(tuple(atom_type))


def bond_features_v1(bond, **kwargs):
    """Return an integer hash representing the bond type.

    flipped : bool
        Only valid for 'v3' version, whether to swap the begin and end atom
        types

    """

    return str(
        (
            bond.GetBondType(),
            bond.GetIsConjugated(),
            bond.IsInRing(),
            sorted([bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]),
        )
    )


def bond_features_v2(bond, **kwargs):
    return str(
        (
            bond.GetBondType(),
            bond.GetIsConjugated(),
            bond.GetStereo(),
            get_ring_size(bond),
            sorted([bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()]),
        )
    )


def bond_features_v3(bond, flipped=False):
    if not flipped:
        start_atom = atom_features_v1(bond.GetBeginAtom())
        end_atom = atom_features_v1(bond.GetEndAtom())

    else:
        start_atom = atom_features_v1(bond.GetEndAtom())
        end_atom = atom_features_v1(bond.GetBeginAtom())

    return str(
        (
            bond.GetBondType(),
            bond.GetIsConjugated(),
            bond.GetStereo(),
            get_ring_size(bond),
            bond.GetEndAtom().GetSymbol(),
            start_atom,
            end_atom,
        )
    )


def bond_features_wbo(start_atom, end_atom, bondatoms):

    start_atom_symbol = bondatoms[0].GetSymbol()
    end_atom_symbol = bondatoms[1].GetSymbol()

    return str((start_atom_symbol, end_atom_symbol))
