import requests
import json
import os

def get_structure(seq, name):
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    response = requests.post(url, data=seq)
    if response.status_code == 200:
        pdb_content = response.text
        filename = f"{name}_predicted.pdb"
        with open(filename, "w") as f:
            f.write(pdb_content)
        print(f"✅ Structure for {name} saved to {filename}")
        return filename
    else:
        print(f"❌ Error: {response.status_code}")
        return None

# Ricin wildtype sequence
ricin_seq = "MKPGGNTIVIWMYAVATWLCFGSTSGWSFTLEDNNIFPKQYPIINFTTAGATVQSYTNFIRAVRGRLTTGADVRHEIPVLPNRVGLPINQRFILVELSNHAELSVTLALDVTNAYVVGYRAGNSAYFFHPDNQEDAEAITHLFTDVQNRYTFAFGGNYDRLEQLAGNLRENIELGNGPLEEAISALYYYSTGGTQL"

get_structure(ricin_seq, "ricin_wt")