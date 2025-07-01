# -*- coding: utf-8 -*-
"""
Created on Thu May 29 19:02:31 2025

@email: Matthias.Schoeffel@lmu.de
"""

from io import open
from conllu import parse_incr
import pandas as pd
from pathlib import Path
from conllu import parse

dateiname="naf.conllu" 

f=Path(dateiname) 
file_name_without_ext = f.stem

data_file = open(f, "r", encoding="utf-8") 

form=[]
pos=[]
for tokenlist in parse_incr(data_file):
    for token in tokenlist:
        pos.append(token["upos"])
        form.append(token["form"])


d={"Lemma":form, "POS":pos}
df = pd.DataFrame(d)

df.to_excel(file_name_without_ext+'.xlsx', index=False)