"""Download the Jena Climate dataset (auth-free) into ./data/."""
import os, io, zipfile, urllib.request

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
DEST = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DEST, exist_ok=True)

print("Downloading Jena Climate dataset ...")
with urllib.request.urlopen(URL) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
    z.extractall(DEST)
print("Saved to", os.path.join(DEST, "jena_climate_2009_2016.csv"))
