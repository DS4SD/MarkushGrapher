import argparse
import json
import re
from pathlib import Path

from rdkit import Chem

ABBREVIATIONS = {
    "iPr": {"smiles": ["*C(C)C"], "population": 141758},
    "CO2H": {"smiles": ["*C(=O)O"], "population": 1092277},
}


class Abbreviation:

    def __init__(self, abbreviations: dict[str, dict]) -> None:
        self.ABBREVIATIONS = {}
        for abb, data in abbreviations.items():
            smi = data.get("smiles", [])[0]
            # skip abbreviation with more than 1 connection.
            if smi.count("*") > 1:
                continue
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if mol is None:
                print(f"PROBLEM with {abb=} {smi=}")
                continue
            att_pts = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == "*":
                    if len(atom.GetNeighbors()) != 1:
                        continue
                    att_pts.append(
                        {"del": atom.GetIdx(), "to": atom.GetNeighbors()[0].GetIdx()}
                    )
                #
            #
            self.ABBREVIATIONS[abb] = {"smi": smi, "mol": mol, "att_pts": att_pts}
        #

    def _cxs_to_smi(self, cxsmiles: str) -> str:
        return re.sub(r" \|(.*)\|", r"", cxsmiles)

    def _cxs_to_grps(self, cxsmiles: str) -> list[str]:
        m = re.search(r"(?:[|,])\$((?:.*?;)*.*?)\$(?=[|,])", cxsmiles)
        grps: list = []
        if m is not None:
            grps = m.group(1).split(";")
        return grps

    def _cxs_to_srus(self, cxsmiles: str) -> dict:
        srus = {}
        for i, m in enumerate(
            re.finditer(
                r"(?:[|,])(Sg:n:\d+(?:\,\d+)*:"
                r"([\w\.\'\<\>\= \-\+\(\)\%\*,]*)"
                r":ht(?::\w*:\w*:)?)"
                r"(?=[|,])",
                cxsmiles,
            )
        ):
            gs = m.group(1).split(":")
            srus[i] = {
                "atom_ids": sorted([int(i) for i in gs[2].split(",")]),
                "n": m.group(2),
                "orig": m.group(1),
            }
        return srus

    def _cxs_to_pvgs(self, cxsmiles: str) -> dict:
        """
        Extract PVGs from CXSMILES

        Expand compact pvgs:
          "m:a:a.a,b:b.b,c:c.c,m:d:d.d,e:e.e"
        to
          "m:a:a.a,m:b:b.b,m:c:c.c,m:d:d.d,m:e:e.e"
        """
        pvgs = {}

        for _ in range(1000):
            m = re.search(
                r"(?:[|,])((?:m:)(\d+:\d+(?:\.\d+)*)(,(\d+:\d+(?:\.\d+)*))+)(?=[|,])",
                cxsmiles,
            )
            if m is None:
                break
            #
            cxsmiles = (
                cxsmiles[: m.start(1)]
                + m.group(1).replace(",", ",m:")
                + cxsmiles[m.end(1) :]
            )

        for i, m in enumerate(
            re.finditer(r"(?:[|,])(m:(\d+):((?:\d+\.)*\d+))(?=[|,])", cxsmiles)
        ):
            gs = m.group(3).split(".")
            pvgs[i] = {
                "atom_ids": sorted([int(i) for i in gs]),
                "endpt": int(m.group(2)),
                "orig": m.group(1),
            }
        return pvgs

    def _reorder_srus(self, srus: dict, mol_to_smi_ids: dict, conn: dict) -> None:
        for sru in srus.values():
            atom_ids = []
            for mol_id in sru.get("atom_ids", []):
                smi_id = mol_to_smi_ids.get(mol_id)
                if smi_id is None:
                    # need to add group ids if needed
                    for mol_id2 in conn.get(mol_id, []):
                        smi_id2 = mol_to_smi_ids.get(mol_id2)
                        if smi_id2 is not None:
                            atom_ids.append(smi_id2)
                        #
                    continue
                atom_ids.append(smi_id)
            #
            sru["atom_ids"] = atom_ids
            sru["orig"] = (
                "Sg:n:" + ",".join([str(i) for i in atom_ids]) + ":" + sru["n"] + ":ht"
            )
            #
        #
        return

    def _reorder_pvgs(self, pvgs: dict, mol_to_smi_ids: dict) -> None:
        for pvg in pvgs.values():
            atom_ids = []
            pvg["endpt"] = mol_to_smi_ids.get(pvg["endpt"])
            for mol_id in pvg.get("atom_ids", []):
                smi_id = mol_to_smi_ids.get(mol_id)
                if smi_id is not None:
                    atom_ids.append(smi_id)
                #
            #
            pvg["atom_ids"] = atom_ids
            pvg["orig"] = (
                "m:" + str(pvg["endpt"]) + ":" + ".".join([str(i) for i in atom_ids])
            )
            #
        #
        return

    def _reorder_grps(self, groups: list, mol_to_smi_ids: dict) -> list:
        n_atoms = len(mol_to_smi_ids) - list(mol_to_smi_ids.values()).count(None)
        clean_groups = [""] * n_atoms
        for mol_id, group in enumerate(groups):
            smi_id = mol_to_smi_ids.get(mol_id)
            if smi_id is None:
                continue
            clean_groups[smi_id] = group
        #
        return clean_groups

    def _assemble_cxsmiles(
        self, smiles: str, groups: list, srus: dict, pvgs: dict
    ) -> str:
        exts = []
        if any([len(_) > 0 for _ in groups]):
            exts.append("$" + ";".join(groups) + "$")

        if len(srus) > 0:
            exts.append(",".join([s.get("orig") for s in srus.values()]))

        if len(pvgs) > 0:
            exts.append(",".join([p.get("orig") for p in pvgs.values()]))

        return f"{smiles} |" + ",".join(exts) + "|"

    def expand(self, cxsmiles: str) -> str:
        smiles = self._cxs_to_smi(cxsmiles)
        groups = self._cxs_to_grps(cxsmiles)
        srus = self._cxs_to_srus(cxsmiles)
        pvgs = self._cxs_to_pvgs(cxsmiles)

        mol = Chem.MolFromSmiles(smiles, sanitize=False)

        # in case groups doesnt have the right length, pad it.
        if len(groups) < mol.GetNumAtoms():
            groups.extend([""] * (mol.GetNumAtoms() - len(groups)))
        elif len(groups) > mol.GetNumAtoms():
            groups = groups[0 : mol.GetNumAtoms()]
        #

        # combime
        conn_pts = []
        conn = {}
        n_atoms = mol.GetNumAtoms()
        for ipos, group in enumerate(groups):

            if group == "":
                continue

            # abb_smi = self.ABBREVIATIONS.get(group)
            # if abb_smi is None:
            #    continue
            abb_data = self.ABBREVIATIONS.get(group)
            if abb_data is None:
                continue

            # abb_smi = abb_data.get("smi").replace("*", "")
            # abb_mol = Chem.MolFromSmiles(abb_smi, sanitize=False)
            abb_mol = abb_data.get("mol")

            mol = Chem.CombineMols(mol, abb_mol)

            atom = mol.GetAtomWithIdx(ipos)
            nbor_ids = [a.GetIdx() for a in atom.GetNeighbors()]

            if len(nbor_ids) == 0:
                continue

            conn[ipos] = list(
                range(
                    n_atoms, n_atoms + abb_mol.GetNumAtoms()  # type: ignore[union-attr]
                )
            )
            for abb_att_pt in abb_data.get("att_pts", []):
                conn_pts.append(
                    {
                        "from": nbor_ids[0],
                        "to": abb_att_pt.get("to") + n_atoms,
                        "del_from": ipos,
                        "del_to": abb_att_pt.get("del") + n_atoms,
                    }
                )
            #
            n_atoms += abb_mol.GetNumAtoms()  # type: ignore[union-attr]
        #

        # connect
        rw = Chem.RWMol(mol)
        rm_atom_ids = []
        for conn_pt in conn_pts:
            beg_pt = conn_pt.get("from", -1)
            end_pt = conn_pt.get("to", -1)
            del_beg_pt = conn_pt.get("del_from", -1)
            del_end_pt = conn_pt.get("del_to", -1)

            rw.AddBond(beg_pt, end_pt, Chem.BondType.SINGLE)

            rw.RemoveBond(beg_pt, del_beg_pt)
            rw.GetAtomWithIdx(del_beg_pt).SetAtomicNum(118)
            rw.GetAtomWithIdx(del_beg_pt).SetIsotope(118)
            rm_atom_ids.append(del_beg_pt)

            rw.RemoveBond(end_pt, del_end_pt)
            rw.GetAtomWithIdx(del_end_pt).SetAtomicNum(118)
            rw.GetAtomWithIdx(del_end_pt).SetIsotope(118)
            rm_atom_ids.append(del_end_pt)
            #
        #
        mol = rw.GetMol()

        # rebuild cxsmiles
        Chem.MolToSmiles(mol)
        smilesAtomOutputOrder = json.loads(
            mol.GetProp("_smilesAtomOutputOrder").replace(",]", "]")
        )

        # mapping
        mol_to_smi_ids: dict = {}
        ipos = 0
        for smi_id, mol_id in enumerate(smilesAtomOutputOrder):
            if mol_id in rm_atom_ids:
                mol_to_smi_ids[mol_id] = None
                continue
            mol_to_smi_ids[mol_id] = ipos
            ipos += 1
        #

        # reorder groups
        clean_groups = self._reorder_grps(groups, mol_to_smi_ids)

        # reorder Sg
        self._reorder_srus(srus, mol_to_smi_ids, conn)

        # reorder m
        self._reorder_pvgs(pvgs, mol_to_smi_ids)

        # cleanup smiles
        clean_smiles = Chem.MolToSmiles(mol)
        clean_smiles = re.sub(r"\[118Og\]\.", "", clean_smiles)
        clean_smiles = re.sub(r"\.\[118Og\]", "", clean_smiles)

        return self._assemble_cxsmiles(clean_smiles, clean_groups, srus, pvgs)


def parse_args():
    parser = argparse.ArgumentParser(description="Process a directory of images")

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the directory containing images",
    )

    parser.add_argument(
        "--abb_file", type=Path, required=True, help="abbreviation file"
    )

    return parser.parse_args()


def main():
    import pickle

    """
 1  23 4 5 678 9012 345 6 78 901 2  34 5  67  8  9   0
[1*]CC(C)c1ccc(SCCC2CCN(c3nc4ccc(Cl)cc4s3)CC2)c([2*])c1
    |$CO2H;;;;;;;;;;;;;;;;;;;;;;;;;;;;Q;;$,Sg:n:28:q:ht|
 1  2 34 5 6 78  9 0 123 4567 890 1 23 456 7  89 0  12
[2*]c1cc(C(C)CC(=O)O)ccc1SCCC1CCN(c2nc3ccc(Cl)cc3s2)CC1
    |$Q;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;$,Sg:n:0:q:ht|
    """

    args = parse_args()
    with open(args.abb_file, "r") as fid:
        abbs = json.load(fid)
    #
    with open(args.dataset_dir, "rb") as fid:
        smis = pickle.load(fid)

    abb = Abbreviation(abbs)

    for smi in smis:
        smi_exp = None
        if smi is not None:
            smi_exp = abb.expand(smi)
        print(json.dumps({"original": smi, "expanded": smi_exp}))


if __name__ == "__main__":
    main()
