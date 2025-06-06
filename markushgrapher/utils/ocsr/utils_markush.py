#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
from collections import defaultdict

from matplotlib import colormaps
from PIL import Image, ImageOps
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
from markushgenerator.cxsmiles_tokenizer import CXSMILESTokenizer
from rdkit.Chem.Draw import rdMolDraw2D


def get_molecule_from_smiles(smiles, remove_stereochemistry, kekulize=True):
    parser_params = Chem.SmilesParserParams()
    parser_params.strictCXSMILES = False
    parser_params.sanitize = False
    parser_params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, parser_params)
    if molecule is None:
        return None
    if remove_stereochemistry:
        Chem.RemoveStereochemistry(molecule)
    # molecule.UpdatePropertyCache(strict=False)
    if kekulize:
        sanity = Chem.SanitizeMol(
            molecule,
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_KEKULIZE
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
            catchErrors=True,
        )
    else:
        sanity = Chem.SanitizeMol(
            molecule,
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
            catchErrors=True,
        )
    if sanity != Chem.rdmolops.SANITIZE_NONE:
        return None
    return molecule


def canonicalize_markush(cxsmiles, verbose=False):
    """
    Warning: The method can convert R-group fragments from C[*] to [*]C, making the new cxsmiles incompatible with .convert_cdk_to_opt().
    However, this can be overcame by provided the original molecule (as MOL file) to .convert_cdk_to_opt().
    """
    parser_params = Chem.SmilesParserParams()
    parser_params.strictCXSMILES = False
    parser_params.sanitize = False
    parser_params.removeHs = False
    molecule = Chem.MolFromSmiles(cxsmiles, parser_params)

    # Get canonical CXSMILES
    canonical_cxsmiles = Chem.MolToCXSmiles(molecule)
    tmp_cxsmiles = Chem.MolToSmiles(molecule)
    if verbose:
        print(f"Converted back CXSMILES: {canonical_cxsmiles}")
    smi_to_smicanon_i_mapping = {
        k: v
        for k, v in zip(
            list(map(int, molecule.GetProp("_smilesAtomOutputOrder")[1:-2].split(","))),
            range(0, molecule.GetNumAtoms()),
        )
    }
    if len(cxsmiles.split("|")) == 1:
        return canonical_cxsmiles

    # Copy m section, updating atom indices
    cxsmiles_tokenizer = CXSMILESTokenizer()
    sections = cxsmiles_tokenizer.parse_sections(cxsmiles.split("|")[1])
    new_sections = []
    for section in sections:
        if (len(section) >= 1) and not (section[0] == "m"):
            continue
        m_section = cxsmiles_tokenizer.parse_m_section(section)
        atom_connector_idx = m_section[1]
        ring_atoms_indices = [idx for idx in m_section[2:] if idx != "."]
        if not (int(atom_connector_idx) in smi_to_smicanon_i_mapping) or any(
            not (int(i) in smi_to_smicanon_i_mapping) for i in ring_atoms_indices
        ):
            # Can be triggerred if the prediction uses indices not present in the original molecule
            print(f"Error in canonicalize_markush for: {cxsmiles}")
            return None
        new_section = f"m:{smi_to_smicanon_i_mapping[int(atom_connector_idx)]}:{'.'.join(str(smi_to_smicanon_i_mapping[int(i)]) for i in ring_atoms_indices)}"
        new_sections.append(new_section)

    if len(canonical_cxsmiles.split("|")) > 1:
        canonical_cxsmiles = (
            canonical_cxsmiles[:-1] + "," + ",".join(new_sections) + "|"
        )
    else:
        canonical_cxsmiles = canonical_cxsmiles + " |" + ",".join(new_sections) + "|"
    return canonical_cxsmiles


def display_markush(
    cxsmiles,
    image_size=[750, 750],
    rdkit_resolution=[750, 750],
    alpha=1,
    force_molecule_sanitization=False,
    verbose=False,
):
    """
    Using the version rdkit==2024.3.5 seems mandatory.
    """
    cxsmiles_tokenizer = CXSMILESTokenizer()
    colors = colormaps["Accent"].colors
    colors = [(color[0], color[1], color[2], alpha) for color in colors]

    # Read molecule
    parser_params = Chem.SmilesParserParams()
    parser_params.strictCXSMILES = False
    parser_params.sanitize = False
    parser_params.removeHs = False
    m = Chem.MolFromSmiles(cxsmiles, parser_params)

    if len(cxsmiles.split("|")) > 1:
        # Detect invalid indices predicted (lead to segmentation fault when calling DrawMoleculeWithHighlights())
        sections_validity = []
        for i, section in enumerate(
            cxsmiles_tokenizer.parse_sections(cxsmiles.split("|")[1])
        ):
            sections_validity.append(True)
            if (len(section) >= 1) and (section[0] == "m"):
                m_section = cxsmiles_tokenizer.parse_m_section(section)
                atom_connector_idx = m_section[1]
                ring_atoms_indices = [idx for idx in m_section[2:] if idx != "."]
                for idx in [atom_connector_idx] + ring_atoms_indices:
                    if not (int(idx) in range(m.GetNumAtoms())):
                        sections_validity[i] = False
                continue
            if (len(section) >= 2) and (section[:2] == "Sg"):
                sg_section = cxsmiles_tokenizer.parse_sg_section(section)
                indices = []
                for idx in sg_section[2:]:
                    if idx == ",":
                        continue
                    if idx == "<atom_list_end>":
                        break
                    indices.append(idx)
                for idx in indices:
                    if not (int(idx) in range(m.GetNumAtoms())):
                        sections_validity[i] = False
        if any([(v == False) for v in sections_validity]):
            print("Invalid CXSMILES for display (index out of range)")
            # Erase a section if it is invalid
            for i, section in enumerate(
                cxsmiles_tokenizer.parse_sections(cxsmiles.split("|")[1])
            ):
                if sections_validity[i] == False:
                    cxsmiles = cxsmiles.replace(section, "")

    # Read molecule
    parser_params = Chem.SmilesParserParams()
    parser_params.strictCXSMILES = False
    parser_params.sanitize = False
    parser_params.removeHs = False
    m = Chem.MolFromSmiles(cxsmiles, parser_params)

    # Convert [d*] to C to display r-labels correctly, even on connected fragments
    for i, atom in enumerate(m.GetAtoms()):
        if atom.GetSymbol() == "*":
            atom.SetAtomicNum(6)

    # Display atom indices
    for i, atom in enumerate(m.GetAtoms()):
        atom.SetProp("atomNote", str(atom.GetIdx()))

    # Read M sections
    m_sections = {}
    if len(cxsmiles.split("|")) > 1:
        sections = cxsmiles_tokenizer.parse_sections(cxsmiles.split("|")[1])
        for section in sections:
            if (len(section) >= 1) and (section[0] == "m"):
                m_section = cxsmiles_tokenizer.parse_m_section(section)
                atom_connector = m_section[1]
                ring_atoms = [idx for idx in m_section[2:] if idx != "."]
                m_sections[atom_connector] = [int(c) for c in ring_atoms]

    # Highlight M sections
    color_idx = 0
    atoms_highlights = defaultdict(list)
    for i, (atom_connector, atoms_ring) in enumerate(m_sections.items()):
        color = colors[color_idx]
        atoms_highlights[int(atom_connector)].append(color)
        for atom_ring in atoms_ring:
            atoms_highlights[int(atom_ring)].append(color)
        color_idx = (color_idx + 1) % len(colors)

    # Highlight Sg sections
    for sgroup in Chem.rdchem.GetMolSubstanceGroups(m):
        color = colors[color_idx]
        for atom_idx in sgroup.GetAtoms():
            atoms_highlights[atom_idx].append(color)
        for bond_idx in sgroup.GetBonds():
            existing_note = "Sg: "
            if m.GetBondWithIdx(bond_idx).HasProp("bondNote"):
                existing_note = m.GetBondWithIdx(bond_idx).GetProp("bondNote")
            m.GetBondWithIdx(bond_idx).SetProp(
                "bondNote", "Sg:" + existing_note[3:] + sgroup.GetProp("LABEL") + " "
            )
        color_idx = (color_idx + 1) % len(colors)

    if force_molecule_sanitization:
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
            ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS,
        )
    # Draw molecule
    d2d = rdMolDraw2D.MolDraw2DCairo(*rdkit_resolution)
    d2d.DrawMoleculeWithHighlights(m, "", dict(atoms_highlights), {}, {}, {})
    d2d.FinishDrawing()
    image = Image.open(io.BytesIO(d2d.GetDrawingText())).convert("RGB")
    image = ImageOps.contain(image, image_size)
    return image
