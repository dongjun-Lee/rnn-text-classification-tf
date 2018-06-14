import wget
import os
import zipfile


glove_dir = "glove"
glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"

if not os.path.exists(glove_dir):
    os.mkdir(glove_dir)

# Download glove vector
wget.download(glove_url, out=glove_dir)

# Extract glove file
with zipfile.ZipFile(os.path.join("glove", "glove.6B.zip"), "r") as z:
    z.extractall(glove_dir)
