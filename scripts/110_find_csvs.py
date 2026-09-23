import os
import redivis
import pandas as pd

OUT = "./inspect_files"
os.makedirs(OUT, exist_ok=True)

d = redivis.user("aimi").dataset("inspect:2n96")
d.get()

for tn in ["full", "sample"]:
    print("")
    print("#" * 55)
    print("TABLE:", tn)
    t = d.table(tn)
    df = t.to_pandas_dataframe(
        variables=["file_id", "file_name",
                   "size"],
        dtype_backend="numpy")
    print("rows:", len(df))

    df["dir"] = df["file_name"].str.rsplit(
        "/", n=1).str[0]
    print("")
    print("directories:")
    print(df["dir"].value_counts().to_string())

    nz = df[~df["file_name"].str.lower()
            .str.endswith(".nii.gz")]
    print("")
    print("non-NIfTI files:", len(nz))
    print(nz[["file_name", "size"]]
          .to_string())

    for _, r in nz.iterrows():
        if r["size"] > 500_000_000:
            continue
        fid = r["file_id"]
        nm = os.path.basename(r["file_name"])
        try:
            f = redivis.file(fid)
            f.download(path=os.path.join(OUT, nm),
                       overwrite=True)
            print("downloaded", nm)
        except Exception as e:
            print("fail", nm, e)

