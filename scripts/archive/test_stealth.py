from Bio.Blast import NCBIWWW

# Sample 5 - 28.5% recovery
stealth_seq = "KAPSYSDLVLSTMVVQQWWLAEPFVPFDKVYLQDFPVPSKYTEISFNFNGATPHTEEQFWLDVKAELFKDGEIVDGIAQFPSEKGLPYERRFLYVSISNSDNYTVTFAVRRTDGMIVGYQAGDVSVWFRPNNEEVAEIIKRLFTQIEQRHQLPYTGEMPVLQRLAGKNGEDCPLGKEPLGACVALLDSWANGSRDP"

print("🕵️ Checking if BLAST is blind to the 28% variant...")
# We use 'blastp' (protein) against 'nr' (non-redundant database)
result_handle = NCBIWWW.qblast("blastp", "nr", stealth_seq)

with open("stealth_blast_results.xml", "w") as out_handle:
    out_handle.write(result_handle.read())
result_handle.close()
print("✅ Results saved. Run: grep 'Hit_def' stealth_blast_results.xml")