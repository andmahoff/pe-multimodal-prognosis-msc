import os
import redivis

print("token set:",
      bool(os.environ.get("REDIVIS_API_TOKEN")))

ds = redivis.user("shahlab").dataset(
    "inspect_ehr:dzc6:v1_2"
)
print("dataset:", ds.name)

for t in ds.list_tables():
    p = t.properties
    print("%-40s rows=%-12s bytes=%s"
          % (t.name,
             p.get("numRows"),
             p.get("numBytes")))

