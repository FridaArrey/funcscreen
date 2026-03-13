from Bio import Entrez, SeqIO

Entrez.email = "frida.arreytakubetang@gmail.com"

handle = Entrez.efetch(db="protein", id="P02879", rettype="fasta", retmode="text")
record = SeqIO.read(handle, "fasta")

print(record.id)
print(record.seq)
print(f"Length: {len(record.seq)} amino acids")

with open("ricin_wildtype.fasta", "w") as f:
    f.write(f">{record.id}\n{record.seq}\n")
    
print("Saved to ricin_wildtype.fasta")