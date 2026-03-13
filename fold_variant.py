import requests

def fold_variant(sequence, filename):
    print(f"🚀 Sending sequence to ESMFold API...")
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    
    # Use the first variant sequence from your ProteinMPNN output (Sample 1)
    response = requests.post(url, data=sequence)
    
    if response.status_code == 200:
        with open(filename, "w") as f:
            f.write(response.text)
        print(f"✅ Success! Saved to {filename}")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

# PASTE YOUR SAMPLE 1 SEQUENCE HERE
variant_1_seq = "HMPPLKEIALHVLRIDHALRSHPYEPFVKVFDDSEEEPEEFPTISFSTDGATEESWSAFMREVKAELTKGGKIVNGVPQLPEREGLPKDKRFVKVKLSNKNGKEVTLAIDRSNGRIVGYQAGDTAVFFKPENEREAEDLKTLFTDVKTKETLPFTGEVPVLEEIAGVKVEDLPLGKEPLEEAIDTLYNYATGSKEN"

fold_variant(variant_1_seq, "variant_1_predicted.pdb")