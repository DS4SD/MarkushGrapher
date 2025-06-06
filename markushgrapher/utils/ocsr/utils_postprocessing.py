from rdkit import Chem


class MoleculePostprocessor:
    def __init__(self):
        pass

    def postprocess_aromatic_rings(self, smiles):
        return

    def postprocess(self, smiles):
        # smiles = self.postprocess_aromatic_rings(smiles)
        return smiles


def main():
    molecule_postprocessor = MoleculePostprocessor()
    smiles = "[1*]c1ccc(Nc2nc3[nH][2*]c([3*])c3c(NC3CCCN(C(=O)C=C)C3)n2)cc1 |$R';;;;;;;;;;X1;;R5;;;;;;;;;;;;;;;;;$|"
    postprocessed_smiles = molecule_postprocessor.postprocess()
    print(postprocessed_smiles)
