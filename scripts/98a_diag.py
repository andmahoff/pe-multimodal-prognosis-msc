import pandas as pd

BASE = "."
DATA = BASE + "/fusion_workspace/data"
EXT = (BASE + "/mimic_ext_pe/physionet.org/files"
       "/mimic-iv-ext-pe/1.0.0/MIMIC-IV-Ext-PE.csv")
ADM = DATA + "/index_admission_times.csv"

ext = pd.read_csv(EXT)
print("cols:", list(ext.columns))
print("rows:", len(ext))
print("")
print("charttime dtype:", ext["charttime"].dtype)
print("charttime nulls:",
      int(ext["charttime"].isna().sum()))
print("charttime head:")
print(ext["charttime"].head(5).to_list())
print("")
print("storetime head:")
print(ext["storetime"].head(3).to_list())
print("")
print("note_id head:")
print(ext["note_id"].head(3).to_list())
print("hadm_id nulls:",
      int(ext["hadm_id"].isna().sum()))

p = pd.to_datetime(ext["charttime"],
                   errors="coerce")
print("")
print("parsed ok:", int(p.notna().sum()),
      "of", len(p))
print("parsed range:", p.min(), "->", p.max())

adm = pd.read_csv(ADM)
a = pd.to_datetime(adm["admittime"],
                   errors="coerce")
print("")
print("admittime head:",
      adm["admittime"].head(3).to_list())
print("admittime parsed:",
      int(a.notna().sum()), "of", len(a))
print("admittime range:", a.min(), "->", a.max())

