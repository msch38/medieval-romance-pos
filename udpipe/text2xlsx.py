# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 19:16:46 2025

"""

import pandas as pd

# Load the tab-separated file
df = pd.read_csv("texte_fets.txt", sep='\t')

pos_map = {
    # Nouns
    "N": "NOUN",
    "NPR": "PROPN",
    "NUM": "NUM",

    # Adjectives and Adverbs
    "ADJ": "ADJ",
    "ADJ^Q": "ADJ",
    "ADJ^POS": "ADJ",
    "ADV": "ADV",
    "MAN": "ADV",
    "WADJ": "ADJ",
    "WADV": "ADV",

    # Determiners
    "D": "DET",
    "D^POS": "DET",
    "D^Q": "DET",
    "D^DEM": "DET",
    "DPR": "DET",
    "PRO^D^1^PL": "DET",
    "PRO^D^3^PL": "DET",
    "PRO^D^3^SG": "DET",
    "PRO^D^1^SG": "DET",
    "PRO^D^2^PL": "DET",

    # Pronouns
    "PRO^1^PL": "PRON",
    "PRO^2^PL": "PRON",
    "PRO^3^PL": "PRON",
    "PRO^3^SG": "PRON",
    "PRO^1^SG": "PRON",
    "PRO^2^SG": "PRON",
    "PRO^A^1^PL": "PRON",
    "PRO^A^3^SG": "PRON",
    "PRO^A^3^PL": "PRON",
    "PRO^A^NTR": "PRON",
    "PRO^A^1^SG": "PRON",
    "PRO^A^2^PL": "PRON",
    "PRO^ADV": "ADV",
    "PRO^Q": "PRON",
    "PRO^RFL^3^SG": "PRON",
    "PRO^RFL^3^PL": "PRON",
    "PRO^RFL^2^PL": "PRON",
    "PRO^RFL^1^PL": "PRON",
    "PRO^RFL^1^SG": "PRON",
    "PRO^RFL^2^SG": "PRON",
    "WPRO": "PRON",

    # Verbs
    "VB": "VERB",
    "VN": "VERB",
    "VNI": "VERB",
    "VG": "VERB",
    "VBPI^3^SG": "VERB",
    "VBPI^3^PL": "VERB",
    "VBPI^2^PL": "VERB",
    "VBPI^2^SG": "VERB",
    "VBPI^1^PL": "VERB",
    "VBPI^1^SG": "VERB",
    "VBPS^3^SG": "VERB",
    "VBPS^3^PL": "VERB",
    "VBPS^2^PL": "VERB",
    "VBPS^1^PL": "VERB",
    "VBPS^1^SG": "VERB",
    "VBDS^3^SG": "VERB",
    "VBDS^3^PL": "VERB",
    "VBDS^1^PL": "VERB",
    "VBDS^1^SG": "VERB",
    "VBDS^2^PL": "VERB",
    "VBDS^2^SG": "VERB",
    "VBDI^3^SG": "VERB",
    "VBDI^3^PL": "VERB",
    "VBDI^2^PL": "VERB",
    "VBDI^2^SG": "VERB",
    "VBDI^1^PL": "VERB",
    "VBDI^1^SG": "VERB",
    "VBFI^1^PL": "VERB",
    "VBFI^2^PL": "VERB",
    "VBFI^3^PL": "VERB",
    "VBFI^3^SG": "VERB",
    "VBFI^1^SG": "VERB",
    "VBI^2^PL": "VERB",
    "VBI^2^SG": "VERB",
    "VBI^1^PL": "VERB",
    "VBPC^1^SG": "VERB",
    "VBPC^1^PL": "VERB",
    "VBPC^2^PL": "VERB",
    "VBPC^2^SG": "VERB",
    "VBPC^3^SG": "VERB",
    "VBPC^3^PL": "VERB",

    # Conjunctions
    "C": "SCONJ",
    "CONJ": "CCONJ",
    "CONJ+NEG": "CCONJ",

    # Prepositions & contractions
    "P": "ADP",
    "P+D": "ADP",

    # Negation
    "NEG": "PART",

    # Punctuation & symbols
    ".": "PUNCT",
    ",": "PUNCT",
    ":": "PUNCT",
    ";": "PUNCT",
    "COMMA": "PUNCT",
    "§": "SYM",
    "?": "PUNCT",
    "¿": "PUNCT",
    "]": "PUNCT",
    "-": "SYM",
    "!": "PUNCT",
    "[": "PUNCT",
    "\"": "PUNCT",
    "\\\"": "PUNCT",  # escaped double quote

    # Structural or editorial markers
    "FOL": "SYM",
    "OLB": "SYM",
    "OLE": "SYM",
    "FW": "X",
    "ASIDEB": "X",
    "ASIDEE": "X",

    # Interjections
    "INTJ": "INTJ"
}





# Map the POS column to UPOS
df["UPOS"] = df["POS"].map(pos_map)#.fillna("X")  # Use 'X' for unknowns

# Save result
df.to_excel("output_with_upos.xlsx", index=False)
