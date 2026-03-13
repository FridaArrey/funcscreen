import requests
import io
from Bio import SeqIO

def fetch_protein_data(uniprot_id):
    # Updated URL format for UniProt REST API
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    print(f"🌐 Querying UniProt for: {uniprot_id}...")
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        # Check if the response actually looks like a FASTA (starts with '>')
        if not response.text.startswith(">"):
            print(f"⚠️ Unexpected format for {uniprot_id}. Check the ID.")
            return None
        
        fasta_io = io.StringIO(response.text)
        record = SeqIO.read(fasta_io, "fasta")
        
        return {
            "id": record.id,
            "name": record.name,
            "description": record.description,
            "sequence": str(record.seq)
        }
        
    except Exception as e:
        print(f"❌ Error fetching {uniprot_id}: {e}")
        return None

# Test with a single known ID first
protein_list = ["P69905", "P06751"] # Hemoglobin Alpha and Ricin A

print("--- Starting UniProt Batch Fetch ---")
dataset = []
for p_id in protein_list:
    data = fetch_protein_data(p_id)
    if data:
        dataset.append(data)
        print(f"✅ Retrieved: {data['id']}")

print(f"\n📁 Dataset size: {len(dataset)}")