from Bio.Blast import NCBIWWW, NCBIXML

# Using sample=1 from your output
query_seq = "HMPPLKEIALHVLRIDHALRSHPYEPFVKVFDDSEEEPEEFPTISFSTDGATEESWSAFMREVKAELTKGGKIVNGVPQLPEREGLPKDKRFVKVKLSNKNGKEVTLAIDRSNGRIVGYQAGDTAVFFKPENEREAEDLKTLFTDVKTKETLPFTGEVPVLEEIAGVKVEDLPLGKEPLEEAIDTLYNYATGSKEN"

print("🚀 Running BLAST against NCBI databases (this may take a minute)...")
result_handle = NCBIWWW.qblast("blastp", "nr", query_seq)

with open("blast_results.xml", "w") as out_handle:
    out_handle.write(result_handle.read())
result_handle.close()

print("✅ Results saved to blast_results.xml. Check if 'Ricin' appears in the top hits!")