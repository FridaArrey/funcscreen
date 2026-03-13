import requests

def fold_dud_variant(sequence, filename):
    print(f"🚀 Sending T=2.0 Dud (10.2% recovery) to ESMFold API...")
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    
    response = requests.post(url, data=sequence)
    
    if response.status_code == 200:
        with open(filename, "w") as f:
            f.write(response.text)
        print(f"✅ Success! Dud structure saved to: {filename}")
    else:
        print(f"❌ Error {response.status_code}")

# The T=2.0 "Dud" Sequence
dud_seq = "SQYAGEYLFWRNHGHIHNIINYDWRYHLLFVLETTTWSNMCMPVEWCFTNEWVDWGSPWFADEQAEKGRCGQVQGMRPVNIPKQGTGFSSMFNSDSCCRGGCLCVIFIRSSCNMLEWPGPHGLMRQYPMFTDVESFNRLNQLDTQCSELITFPAQGNAMIRNITTDMGSGKLAAGVFMAMKAVDITEAWKNIAIDY"

fold_dud_variant(dud_seq, "dud_variant_predicted.pdb")