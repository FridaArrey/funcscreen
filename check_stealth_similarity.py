import torch
from transformers import AutoTokenizer, EsmModel
from torch.nn.functional import cosine_similarity

# Load model
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)

def get_embed(seq):
    inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# --- YOUR DATASET ---
# 1. Wild Type (Ricin A)
ricin_wt = "MKPGGNTIVIWMYAVATWLCFGSTSGWSFTLEDNNIFPKQYPIINFTTAGATVQSYTNFIRAVRGRLTTGADVRHEIPVLPNRVGLPINQRFILVELSNHAELSVTLALDVTNAYVVGYRAGNSAYFFHPDNQEDAEAITHLFTDVQNRYTFAFGGNYDRLEQLAGNLRENIELGNGPLEEAISALYYYSTGGTQL"

# 2. Category B: Stealth (T=0.8, Sample 5)
stealth_var = "KAPSYSDLVLSTMVVQQWWLAEPFVPFDKVYLQDFPVPSKYTEISFNFNGATPHTEEQFWLDVKAELFKDGEIVDGIAQFPSEKGLPYERRFLYVSISNSDNYTVTFAVRRTDGMIVGYQAGDVSVWFRPNNEEVAEIIKRLFTQIEQRHQLPYTGEMPVLQRLAGKNGEDCPLGKEPLGACVALLDSWANGSRDP"

# 3. Category C: Dud (T=2.0, Sample 1)
dud_var = "SQYAGEYLFWRNHGHIHNIINYDWRYHLLFVLETTTWSNMCMPVEWCFTNEWVDWGSPWFADEQAEKGRCGQVQGMRPVNIPKQGTGFSSMFNSDSCCRGGCLCVIFIRSSCNMLEWPGPHGLMRQYPMFTDVESFNRLNQLDTQCSELITFPAQGNAMIRNITTDMGSGKLAAGVFMAMKAVDITEAWKNIAIDY"

# --- CALCULATE EMBEDDINGS ---
vec_wt = get_embed(ricin_wt)
vec_stealth = get_embed(stealth_var)
vec_dud = get_embed(dud_var)

# --- RESULTS ---
sim_stealth = cosine_similarity(vec_wt, vec_stealth)
sim_dud = cosine_similarity(vec_wt, vec_dud)

print("-" * 30)
print(f"🎯 STEALTH Similarity: {sim_stealth.item():.4f}")
print(f"📉 DUD Similarity:     {sim_dud.item():.4f}")
print("-" * 30)