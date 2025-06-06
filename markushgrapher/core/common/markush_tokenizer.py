import json
import os
import re
from ast import literal_eval

from rdkit import Chem
from SmilesPE.pretokenizer import atomwise_tokenizer


class MarkushTokenizer:
    def __init__(
        self,
        tokenizer,
        dataset_path,
        encode_position=False,
        grounded_smiles=False,
        training_dataset_name=None,
        encode_index=False,
        condense_labels=True,
    ):
        self.rtable_item_separator = "<ns>"
        if training_dataset_name == "mdu_300":
            self.substituents_separator = ","
        else:
            self.substituents_separator = "<n>"
        print(f"Substituents separator is: {self.substituents_separator}")
        self.tokenizer = tokenizer
        if training_dataset_name:
            self.training_dataset_name = training_dataset_name
            print(f"self.training_dataset_name: {training_dataset_name}")
        else:
            self.training_dataset_name = dataset_path.split("/")[-1]
            print(
                f"self.training_dataset_name (automatically set): {self.training_dataset_name}"
            )
        self.encode_position = encode_position
        self.grounded_smiles = grounded_smiles
        self.condense_labels = condense_labels
        print("self.condense_labels", self.condense_labels)
        self.encode_index = encode_index
        self.set_vocabulary()

    def compress_stable(self, stable):
        # Compact lists of variable groups
        value_to_keys = {}
        for key, value in stable.items():
            value_tuple = tuple(value)
            if value_tuple in value_to_keys:
                value_to_keys[value_tuple].append(key)
            else:
                value_to_keys[value_tuple] = [key]
        stable = {}
        for value, keys in value_to_keys.items():
            merged_key = ",".join(keys)
            stable[merged_key] = list(value)

        # Compress lists of integers
        for label, substituents in stable.items():
            are_substituents_integers = False
            substituents_values = []
            for substituent in substituents:
                try:
                    substituents_values.append(int(substituent))
                    are_substituents_integers = True
                except:
                    are_substituents_integers = False
                    continue

            if are_substituents_integers:
                # Test if indices are consecutives
                if substituents_values == list(
                    range(min(substituents_values), max(substituents_values) + 1)
                ):
                    compressed_int_list = (
                        str(min(substituents_values))
                        + "-"
                        + str(max(substituents_values))
                    )
                    stable[label] = compressed_int_list
        return stable

    def get_stable(self, text, verbose=False):
        # Get stable string
        try:
            stable_string = re.search(
                re.escape("<stable>") + r"(.*?)" + re.escape("</stable>"), text
            ).group(1)
        except Exception as e:
            if verbose:
                print(f"Error {e} in get_stable for {text}")
            postprocessed_stable = False
            if not ("</stable>" in text) and not ("</markush>" in text):
                """
                Display incomplete predictions, as:
                    <markush> <cxsmi> [ R ] C . C 1 [ Y ] N 2 C [ Z ] C 1 C 2 . C S O . [ R 3 ] C |m:9:6.5.4.7,m:13:2.3.4.7.8,m:1:4.7.6.5.4 </cxsmi> <stable>
                    R : Alkyl , aryl-oxidel  <ns> R 1 , R 2 , R 3  <ns> Y : O , S , NH , N-alkane , N-aryl , N-acyl  <ns> Z : Hydro , Alkyl , NH , O-alkane ,
                    O-alkyl , O-aryl , S-alkane , N-aryl , N-aryl  <ns> R : Sodium , Alkyl , NH , O-alkane , O-alkane , O-alkane , O-alkane , S-alkane , S-alkane ,
                    N-Aldiminium , N-aryl , N-aryl  <ns> R : Sodium , Alkyl , NH , O-alkane , O-Alkyl , O-aryl , S-alkanl , N-Aldinism , N-aryl , N-aryl  <ns> R : Sodium ,
                    Alkyl , NH , O-alkane , O-alkane , S-alkanl , S-aryl , S-alkanl , N-aryl , N-aryl  <ns> R : Sodium , Alkanl , aryl , O-alkane , S-alkanl , S-alkanl ,
                    S-alkane , N-Aldinism , N-aryl , N-ary
                """
                if ("<stable>" in text) and (len(text.split("<stable>")) >= 2):
                    stable_string = text.split("<stable>")[1]
                    postprocessed_stable = True
            if not (postprocessed_stable):
                return None

        # Get stable
        try:
            stable = {}
            for item in stable_string.split(self.rtable_item_separator):
                if not (len(item.split(":"))) > 1:
                    continue
                substituents = []
                for s in item.split(":")[1].split(self.substituents_separator):
                    if s[0] == " ":
                        s = s[1:]
                    if (len(s) > 0) and (s[-1] == " "):
                        s = s[:-1]
                    substituents.append(s)

                labels = item.split(":")[0].replace(" ", "")
                for label in labels.split(self.substituents_separator):
                    stable[label] = substituents

            # Convert lists of integers (1-10) to lists of integers (1, 2, 3, 4)
            for label, substituents in stable.items():
                sustituents_to_add = []
                sustituents_to_remove = []
                for substituent in substituents:
                    if not ("-" in substituent):
                        continue
                    limits = [
                        (int(a), int(b))
                        for a, b in re.findall("(\d+)-(\d+)", substituent)
                    ]
                    if (len(limits) == 0) or (len(limits) > 1):
                        continue
                    if not (substituent == str(limits[0][0]) + "-" + str(limits[0][1])):
                        continue

                    sustituents_to_remove.append(
                        str(limits[0][0]) + "-" + str(limits[0][1])
                    )
                    sustituents_to_add.extend(
                        [
                            str(value)
                            for value in range(min(limits[0]), max(limits[0]) + 1)
                        ]
                    )
                for substituent in sustituents_to_remove:
                    stable[label].remove(substituent)
                stable[label].extend(sustituents_to_add)

        except Exception as e:
            if verbose:
                print(f"Error {e} in get_stable for {text}")
            return None
        return stable

    def select_vocab_files(self):
        vocabulary_files = []
        if (
            (self.training_dataset_name == "ocxsr_12")
            or (self.training_dataset_name == "ocxsr_17")
            or (self.training_dataset_name == "ocxsr_19")
        ):
            vocabulary_files = [
                os.path.dirname(__file__) + "/../../../data/vocabulary/ocxsr_12.json"
            ]
            atom_vocabulary_file = (
                os.path.dirname(__file__)
                + "/../../../data/vocabulary/ocxsr_12_atoms.json"
            )

        elif (self.training_dataset_name == "ocsr_3") or (
            self.training_dataset_name == "ocsr_test"
        ):
            if self.encode_position and self.grounded_smiles:
                vocabulary_files = [
                    os.path.dirname(__file__)
                    + "/../../../data/vocabulary/ocsr_3_mol.json"
                ]
                atom_vocabulary_file = (
                    os.path.dirname(__file__)
                    + "/../../../data/vocabulary/ocsr_3_atoms.json"
                )
            else:
                vocabulary_files = [
                    os.path.dirname(__file__) + "/../../../data/vocabulary/ocsr_3.json"
                ]  # 04_09_PM_June_22_2024
                atom_vocabulary_file = (
                    os.path.dirname(__file__)
                    + "/../../../data/vocabulary/ocsr_vocab_atoms.json"
                )

        elif self.training_dataset_name == "ocsr_2":
            vocabulary_files = [
                os.path.dirname(__file__)
                + "/../../../data/vocabulary/ocsr_vocab_atoms.json",
                os.path.dirname(__file__)
                + "/../../../data/vocabulary/ocsr_vocab_bonds.json",
                os.path.dirname(__file__)
                + "/../../../data/vocabulary/ocsr_vocab_chars.json",
            ]
            atom_vocabulary_file = (
                os.path.dirname(__file__)
                + "/../../../data/vocabulary/ocsr_vocab_atoms.json"
            )

        elif (
            (self.training_dataset_name == "ocxsr_2")
            or (self.training_dataset_name == "ocxsr_m_2")
            or (self.training_dataset_name == "ocxsr_11")
            or (self.training_dataset_name == "ocxsr_test")
        ):
            vocabulary_files = [
                os.path.dirname(__file__) + "/../../../data/vocabulary/ocxsr_2.json"
            ]
            atom_vocabulary_file = (
                os.path.dirname(__file__)
                + "/../../../data/vocabulary/ocxsr_2_atoms.json"
            )

        # Default (mdu)
        else:
            vocabulary_files = [
                os.path.dirname(__file__) + "/../../../data/vocabulary/ocsr_3.json"
            ]
            atom_vocabulary_file = (
                os.path.dirname(__file__)
                + "/../../../data/vocabulary/ocsr_vocab_atoms.json"
            )

        return vocabulary_files, atom_vocabulary_file

    def set_vocabulary(self):
        first_index = 0  # By default, 200 other tokens are available.
        vocabulary = []

        # Set base tokens
        if "ocsr" in self.training_dataset_name:
            vocabulary.extend(["<smi>", "</smi>"])
            print("SMILES token <smi> added")
        else:
            # Default
            if "mdu_2002" in self.training_dataset_name:
                vocabulary.extend(["<cxsmi>", "</cxsmi>"])
                print("CXSMILES token <cxsmi> added")
            else:
                vocabulary.extend(["<cxsmi>", "</cxsmi>", "<r>", "</r>"])
                print("CXSMILES token <cxsmi> <r> added")

        if "mdu" in self.training_dataset_name:
            vocabulary.extend(
                [
                    "<markush>",
                    "</markush>",
                    "<stable>",
                    "</stable>",
                    self.substituents_separator,
                    self.rtable_item_separator,
                ]
            )
            print(
                f"Markush tokens <markush> <stable> {self.substituents_separator} {self.rtable_item_separator} added"
            )

        if self.encode_index:
            vocabulary.extend(["<i>", "</i>"])
            print("Index token <i> added")

        # Select vocabulary
        vocabulary_files, atom_vocabulary_file = self.select_vocab_files()
        with open(atom_vocabulary_file) as f:
            self.vocabulary_atoms = list(json.load(f).keys())
        for file_name in vocabulary_files:
            with open(file_name) as f:
                vocabulary.extend(json.load(f).keys())

        self.vocabulary = {
            vocabulary[i]: f"<other_{first_index + i}>" for i in range(len(vocabulary))
        }
        self.vocabulary_inverse = {
            f"<other_{first_index + i}>": vocabulary[i] for i in range(len(vocabulary))
        }
        self.max_vocabulary_range = len(self.vocabulary)

    def cap(self, value):
        if value > 500:
            return 500
        if value < 0:
            return 0
        return value

    def get_rdkit_atom_tokens(self, smiles):
        atoms_chars = []
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        for a in mol.GetAtoms():
            atom = Chem.RWMol()
            atom.AddAtom(a)
            atoms_chars.append(Chem.MolToSmiles(atom))
        return atoms_chars

    def encode_smi(self, label):
        output = []
        output.append(self.tokenizer._convert_token_to_id(self.vocabulary["<smi>"]))

        label = label.replace("<smi>", "").replace("</smi>", "")
        atom_boxes = literal_eval(label.split("!")[1])
        smiles = label.split("!")[0]

        i = 0
        for token in atomwise_tokenizer(smiles):
            if token in self.vocabulary:
                output.append(
                    self.tokenizer._convert_token_to_id(self.vocabulary[token])
                )
            else:
                print(f"{token} not found in vocabulary!")
                output.append(self.tokenizer._convert_token_to_id("<unk>"))
            if self.encode_position and (token in self.vocabulary_atoms):
                # Error: the atoms order in "atom_boxes" and in "atomwise_tokenizer" could be different
                x_min_position = self.cap(int(atom_boxes[i][0] * 500 / 1024))
                y_min_position = self.cap(int(atom_boxes[i][1] * 500 / 1024))
                x_max_position = self.cap(int(atom_boxes[i][2] * 500 / 1024))
                y_max_position = self.cap(int(atom_boxes[i][3] * 500 / 1024))
                output.append(
                    self.tokenizer._convert_token_to_id(f"<loc_{x_min_position}>")
                )
                output.append(
                    self.tokenizer._convert_token_to_id(f"<loc_{y_min_position}>")
                )
                output.append(
                    self.tokenizer._convert_token_to_id(f"<loc_{x_max_position}>")
                )
                output.append(
                    self.tokenizer._convert_token_to_id(f"<loc_{y_max_position}>")
                )
                i += 1

        output.append(self.tokenizer._convert_token_to_id(self.vocabulary["</smi>"]))
        return output

    def encode_cxsmi(self, label, verbose=False):
        if verbose:
            print(f"Encode index in encode_cxsmi: {self.encode_index}")
        output = []
        output.append(self.tokenizer._convert_token_to_id(self.vocabulary["<cxsmi>"]))

        label = label.replace("<cxsmi>", "").replace("</cxsmi>", "")
        if verbose:
            print(f"Label: {label}")
        atom_boxes = literal_eval(label.split("!")[1])
        cxsmiles_opt = label.split("!")[0]

        if len(cxsmiles_opt.split("|")) > 1:
            rtable = "|" + cxsmiles_opt.split("|")[1]
        i = 0

        # Debug
        # cxsmiles_opt = "[H]C=C(N)c1cc(NC)<r>Rr</r>[nH]1"
        # cxsmiles_opt = "<r>Z</r>=NSCC(O)<r>G55</r>(O)CSN=O</cxsmi></s>"

        if not ("mdu_2002" in self.training_dataset_name) or not (self.condense_labels):
            # Replace <r> and </r> with "[" and "]" for using atomwise_tokenizer()
            rgroups_starting_indices_shifted = []
            for match in re.finditer(r"(<r>(.*?)</r>)", cxsmiles_opt):
                rgroups_starting_indices_shifted.append(
                    {match.group(2): match.start(1)}
                )
            if verbose:
                print(
                    "rgroups_starting_indices_shifted", rgroups_starting_indices_shifted
                )
            length_adjustment = 0
            rgroups_starting_indices = []
            for rgroup_starting_index_shifted in rgroups_starting_indices_shifted:
                rgroup, original_index = (
                    list(rgroup_starting_index_shifted.keys())[0],
                    list(rgroup_starting_index_shifted.values())[0],
                )
                current_positions = cxsmiles_opt.find(f"<r>{rgroup}</r>")
                updated_index = original_index - length_adjustment
                rgroups_starting_indices.append(updated_index)
                length_adjustment += (3 - 1) + (4 - 1)
            cxsmiles_opt = cxsmiles_opt.replace("<r>", "[").replace("</r>", "]")

        if verbose:
            print("rgroups_starting_indices", rgroups_starting_indices)

        cxsmiles_opt_i = 0
        for token in atomwise_tokenizer(cxsmiles_opt.split("|")[0]):
            if not ("mdu_2002" in self.training_dataset_name) or not (
                self.condense_labels
            ):
                cxsmiles_opt_i += len(token)
                if verbose:
                    print("cxsmiles_opt_i", cxsmiles_opt_i)
                    print("token", token)
                if (cxsmiles_opt_i - len(token)) in rgroups_starting_indices:
                    token = token.replace("[", "<r>").replace("]", "</r>")

            # If token is atom
            if token in self.vocabulary:
                output.append(
                    self.tokenizer._convert_token_to_id(self.vocabulary[token])
                )

                if self.encode_index and (
                    token in self.vocabulary_atoms
                ):  # and (rdkit_atom_tokens != []):
                    # if token == rdkit_atom_tokens[i]:
                    output.append(
                        self.tokenizer._convert_token_to_id(self.vocabulary["<i>"])
                    )
                    output.append(self.tokenizer._convert_token_to_id(str(i)))
                    output.append(
                        self.tokenizer._convert_token_to_id(self.vocabulary["</i>"])
                    )
                    i += 1
            elif ("<r>" in token) and ("</r>" in token):
                output.append(
                    self.tokenizer._convert_token_to_id(self.vocabulary["<r>"])
                )
                for c in token.replace("<r>", "").replace("</r>", ""):
                    # By default sequences encoded with self.tokenizer.encode() start with a space.
                    output.extend(self.tokenizer.encode(c)[:-1])
                output.append(
                    self.tokenizer._convert_token_to_id(self.vocabulary["</r>"])
                )
                if self.encode_index:
                    output.append(
                        self.tokenizer._convert_token_to_id(self.vocabulary["<i>"])
                    )
                    output.append(self.tokenizer._convert_token_to_id(str(i)))
                    output.append(
                        self.tokenizer._convert_token_to_id(self.vocabulary["</i>"])
                    )
                    i += 1
            else:
                if "mdu_2002" in self.training_dataset_name or not (
                    self.condense_labels
                ):
                    for c in token:
                        output.extend(self.tokenizer.encode(c)[:-1])
                    if (
                        "[" in token and self.encode_index
                    ):  # R-label in cxsmiles_opt (such as [R1])
                        output.append(
                            self.tokenizer._convert_token_to_id(self.vocabulary["<i>"])
                        )
                        output.append(self.tokenizer._convert_token_to_id(str(i)))
                        output.append(
                            self.tokenizer._convert_token_to_id(self.vocabulary["</i>"])
                        )
                        i += 1
                else:
                    for c in token:
                        # By default sequences encoded with self.tokenizer.encode() start with a space.
                        output.extend(self.tokenizer.encode(c)[:-1])

        if len(cxsmiles_opt.split("|")) > 1:
            sections = rtable[1:].split(",")
            new_sections = []
            for i in range(len(sections)):
                if sections[i][0] == "m":
                    new_sections.append(sections[i])
                if sections[i][:2] == "Sg":
                    merged_section = sections[i] + ","
                    j = i + 1
                    while (j < len(sections) and sections[j][0] != "m") and (
                        sections[j][:2] != "Sg"
                    ):
                        merged_section += sections[j] + ","
                        j += 1
                    merged_section = merged_section[:-1]
                    new_sections.append(merged_section)

            output.append(self.tokenizer._convert_token_to_id("|"))
            if not (self.condense_labels):
                output.append(self.tokenizer._convert_token_to_id("$"))
                for c in cxsmiles_opt.split("$")[1]:
                    output.append(self.tokenizer._convert_token_to_id(c))
                output.append(self.tokenizer._convert_token_to_id("$"))
                output.append(self.tokenizer._convert_token_to_id(","))
            # Parse R-table
            for section in new_sections:
                if section[0] in "m":  # m:0:15.16.17.18.19.20
                    m = section.split(":")[0]
                    atom_connector = section.split(":")[1]
                    atom_rings = section.split(":")[2].split(".")

                    for c in f"{m}:":
                        # Encode character per character
                        output.append(self.tokenizer._convert_token_to_id(c))
                    output.append(self.tokenizer._convert_token_to_id(atom_connector))
                    output.append(self.tokenizer._convert_token_to_id(":"))
                    for atom_ring in atom_rings:
                        # Encode index per index
                        output.append(
                            self.tokenizer._convert_token_to_id(atom_ring)
                        )  # Do not split atom indices
                        output.append(self.tokenizer._convert_token_to_id("."))
                    output = output[:-1]

                if section[:2] == "Sg":  # Sg:n:11,12:F:ht
                    sg = section.split(":")[0]
                    label = section.split(":")[1]
                    indices = section.split(":")[2].split(",")
                    end = ":" + ":".join(section.split(":")[3:])
                    for c in f"{sg}:{label}:":
                        # Encode character per character
                        output.append(self.tokenizer._convert_token_to_id(c))

                    for index in indices:
                        # Encode index per index
                        output.append(self.tokenizer._convert_token_to_id(index))
                        output.append(self.tokenizer._convert_token_to_id(","))
                    output = output[:-1]

                    for c in end:
                        # Encode character per character
                        output.append(self.tokenizer._convert_token_to_id(c))
                output.append(self.tokenizer._convert_token_to_id(","))
            output = output[:-1]

        output.append(self.tokenizer._convert_token_to_id(self.vocabulary["</cxsmi>"]))
        return output

    def encode_stable(self, label, verbose=False):
        output = []
        output.append(self.tokenizer._convert_token_to_id(self.vocabulary["<stable>"]))
        label = label.replace("<stable>", "").replace("</stable>", "")

        segments = []
        for token in label.split(":"):
            segments.extend(token.split(self.rtable_item_separator))

        for i in range(len(segments) // 2):
            substituent_labels, substituents = segments[2 * i], segments[2 * i + 1]
            for substituent_label in substituent_labels.split(
                self.substituents_separator
            ):
                for c in substituent_label:
                    # Encode character per character
                    output.extend(self.tokenizer.encode(c)[:-1])
                output.append(
                    self.tokenizer._convert_token_to_id(
                        self.vocabulary[self.substituents_separator]
                    )
                )
            output = output[:-1]
            output.extend(self.tokenizer.encode(":")[:-1])
            for substituent in substituents.split(self.substituents_separator):
                # Encode substituent per substituent
                # Example: a hologen atom -> ['▁', 'a', '▁', 'hal', 'ogen', '▁', 'atom']
                output.extend(self.tokenizer.encode(substituent)[:-1])
                if self.substituents_separator == ",":
                    output.extend(self.tokenizer.encode(",")[:-1])
                else:
                    output.append(
                        self.tokenizer._convert_token_to_id(
                            self.vocabulary[self.substituents_separator]
                        )
                    )
            output = output[:-1]
            output.append(
                self.tokenizer._convert_token_to_id(
                    self.vocabulary[self.rtable_item_separator]
                )
            )
        if len(output) > 2:
            output = output[:-1]
        output.append(self.tokenizer._convert_token_to_id(self.vocabulary["</stable>"]))
        return output

    def encode_markush(self, label, verbose=False):
        output = []
        output.append(self.tokenizer._convert_token_to_id(self.vocabulary["<markush>"]))

        # Encode CXSMILES
        cxsmiles_label = (
            "<cxsmi>"
            + re.search(
                re.escape("<cxsmi>") + r"(.*?)" + re.escape("</cxsmi>"), label
            ).group(1)
            + "</cxsmi>"
        )
        output.extend(self.encode_cxsmi(cxsmiles_label))

        # Encode substituents tables
        stable_label = (
            "<stable>"
            + re.search(
                re.escape("<stable>") + r"(.*?)" + re.escape("</stable>"), label
            ).group(1)
            + "</stable>"
        )
        output.extend(self.encode_stable(stable_label))

        output.append(
            self.tokenizer._convert_token_to_id(self.vocabulary["</markush>"])
        )
        return output

    def clean_cxsmiles_spaces(self, input_string):
        pattern = r"(<cxsmi>)(.*?)(</cxsmi>)"

        def replace_underscores(match):
            return f"{match.group(1)}{match.group(2).replace('▁','')}{match.group(3)}"

        result_string = re.sub(pattern, replace_underscores, input_string)
        return result_string

    def decode_plus_decode_other_tokens(
        self, tokens, permissive_parsing=False, verbose=False
    ):
        decoded_tokens = self.tokenizer.convert_ids_to_tokens(tokens)
        if verbose:
            print(f"Decoded tokens: {decoded_tokens}")
            decoded_sequence_raw = ""
            for t in decoded_tokens:
                if t in self.vocabulary_inverse:
                    print(t, self.vocabulary_inverse[t])
                    decoded_sequence_raw += self.vocabulary_inverse[t]
                else:
                    decoded_sequence_raw += t
            print("decoded_sequence_raw:", decoded_sequence_raw)
        output_str = ""
        skip_next = False
        for i_token, token in enumerate(decoded_tokens):
            if skip_next:
                if self.encode_index and (token != (self.vocabulary["</i>"])):
                    continue
            skip_next = False
            if self.encode_index and (self.vocabulary["<i>"] in token):
                skip_next = True
            if self.encode_index and skip_next:
                continue
            if self.encode_index and (self.vocabulary["</i>"] in token):
                continue
            if ("loc" in token) and ("<" in token) and (">" in token):
                continue
            if ("other" in token) and ("<" in token) and (">" in token):
                output_str += self.vocabulary_inverse[token] + " "
            else:
                # Remove "▁" in the predicted sequence
                if token[0] == "▁":
                    token = token[1:]
                if ((i_token + 1) < len(decoded_tokens)) and (
                    ("▁" in decoded_tokens[i_token + 1])
                    or ("other" in decoded_tokens[i_token + 1])
                ):
                    # Add a space if the next token is also a starting sequence, or a SMILES token
                    output_str += token + " "
                else:
                    # If the next token is not a starting sequence, the space is already encoded
                    output_str += token
        return output_str
