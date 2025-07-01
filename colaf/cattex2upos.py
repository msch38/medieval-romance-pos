# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:33:14 2025

"""

import pandas as pd

# Load the Excel file into a pandas DataFrame
df = pd.read_excel('cleaned_file.xlsx')  # Update this with your file path

# Define a dictionary of modifications to be applied
cattex2009_to_upos = {
    # Verbs (VER*)
    'VERinf': 'VERB',    # infinitive verb
    'VERger': 'VERB',    # gerund
    'VERpar': 'VERB',    # participle
    'VERcjg': 'VERB',    # conjugated verb
    'VERaux': 'AUX',     # auxiliary verb
    'VERimp': 'VERB',    # imperative verb
    'VERpcp': 'VERB',    # past participle
    'VERppa': 'VERB',    # past participle (alternative form)
    'VERppe': 'VERB',
    
    # Adjectives (ADJ*)
    'ADJqua': 'ADJ',     # qualitative adjective
    'ADJord': 'ADJ',     # ordinal adjective
    'ADJind': 'ADJ',     # indefinite adjective
    'ADJpos': 'ADJ',     # possessive adjective
    'ADJdem': 'ADJ',     # demonstrative adjective
    'ADJnum': 'ADJ',     # numeral adjective
    'ADJint': 'ADJ',     # interrogative adjective
    'ADJexc': 'ADJ',     # exclamative adjective
    'ADJcar': 'NUM',
    'ADVgen.PROper': "PRON",
    'ADVneg.PROper': 'PRON',
    
    # Nouns (NOM*)
    'NOMcom': 'NOUN',    # common noun
    'NOMpro': 'PROPN',   # proper noun
    
    # Pronouns (PON*)
    'PONper': 'PRON',    # personal pronoun
    'PONdem': 'PRON',    # demonstrative pronoun
    'PONpos': 'PRON',    # possessive pronoun
    'PONrel': 'PRON',    # relative pronoun
    'PONint': 'PRON',    # interrogative pronoun
    'PONexc': 'PRON',    # exclamative pronoun
    'PONind': 'PRON',    # indefinite pronoun
    'PONfrt': 'PRON',    # reinforcement pronoun / strong pronoun
    'PONfbl': 'PRON',    # weak pronoun / clitic pronoun
    'PONref': 'PRON',    # reflexive pronoun
    
    # Pronouns (PRO*) - alternative format
    'PROper': 'PRON',    # personal pronoun (alternative format)
    'PROdem': 'PRON',    # demonstrative pronoun (alternative format)
    'PROpos': 'PRON',    # possessive pronoun (alternative format)
    'PROrel': 'PRON',    # relative pronoun (alternative format)
    'PROint': 'PRON',    # interrogative pronoun (alternative format)
    'PROexc': 'PRON',    # exclamative pronoun (alternative format)
    'PROind': 'PRON',    # indefinite pronoun (alternative format)
    'PROadv': 'PRON',
    'PROord': 'ADJ',
    'PROcar': 'NUM',
    
    # Determiners (DET*)
    'DETart': 'DET',     # article
    'DETdem': 'DET',     # demonstrative determiner
    'DETpos': 'DET',     # possessive determiner
    'DETind': 'DET',     # indefinite determiner
    'DETnum': 'DET',     # numeral determiner
    'DETint': 'DET',     # interrogative determiner
    'DETexc': 'DET',     # exclamative determiner
    'DETdef': 'DET',     # definite determiner/article
    'DETndf': 'DET',     # indefinite determiner (alternative form)
    'DETcar': 'DET',
    'DETrel': 'DET',
    
    # Adverbs (ADV*)
    'ADVgen': 'ADV',     # general adverb
    'ADVneg': 'ADV',     # negative adverb
    'ADVloc': 'ADV',     # locative adverb
    'ADVtem': 'ADV',     # temporal adverb
    'ADVmod': 'ADV',     # modal adverb
    'ADVcan': 'ADV',     # quantifying adverb
    'ADVint': 'ADV',     # interrogative adverb
    'ADVexc': 'ADV',     # exclamative adverb
    'ADVrel': 'ADV',     # relative adverb
    
    # Prepositions (PRE*)
    'PREsim': 'ADP',     # simple preposition
    'PREcom': 'ADP',     # compound preposition
    'PRE.DETdef': "ADP",
    'PRE': "ADP",
    
    # Conjunctions (CON*)
    'CONcoo': 'CCONJ',   # coordinating conjunction
    'CONsub': 'SCONJ',   # subordinating conjunction
    
    # Numerals (NUM*)
    'NUMcar': 'NUM',     # cardinal numeral
    'NUMord': 'NUM',     # ordinal numeral
    'NUMfra': 'NUM',     # fractional numeral
    'NUMmul': 'NUM',     # multiplicative numeral
    
    # Interjections (INT*)
    'INTprp': 'INTJ',    # proper interjection
    'INTimp': 'INTJ',    # improper interjection
    
    # Punctuation (PUN*)
    'PUNpnt': 'PUNCT',   # period
    'PUNcom': 'PUNCT',   # comma
    'PUNsem': 'PUNCT',   # semicolon
    'PUNcol': 'PUNCT',   # colon
    'PUNint': 'PUNCT',   # question mark
    'PUNexc': 'PUNCT',   # exclamation mark
    'PUNsus': 'PUNCT',   # suspension points
    'PUNpar': 'PUNCT',   # parentheses
    'PUNquo': 'PUNCT',   # quotation marks
    'PUNdas': 'PUNCT',   # dash
    'PUNsla': 'PUNCT',   # slash
    'PUNoth': 'PUNCT',   # other punctuation
    
    # Particles (PAR*)
    'PARneg': 'PART',    # negative particle
    'PARaff': 'PART',    # affirmative particle
    'PARint': 'PART',    # interrogative particle
    
    # Symbols (SIM*)
    'SIMmon': 'SYM',     # monetary symbol
    'SIMmat': 'SYM',     # mathematical symbol
    'SIMoth': 'SYM',     # other symbols
    
    # Foreign/Unknown (EXT*)
    'EXText': 'X',       # foreign text
    'EXTunk': 'X',       # unknown
    'OUT': "VERB"
}

# Replace the words in the 'pos' column according to the dictionary
df['pos'] = df['pos'].replace(cattex2009_to_upos)

# Save the modified DataFrame to a new Excel file
df.to_excel('cleaned_file_mod.xlsx', index=False)
