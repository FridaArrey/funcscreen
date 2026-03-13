import xml.etree.ElementTree as ET

# Parse your existing BLAST results
tree = ET.parse('stealth_blast_results.xml')
root = tree.getroot()

print("📋 BLAST SIGNIFICANCE ANALYSIS")
for hit in root.findall('.//Hit'):
    hit_def = hit.find('Hit_def').text
    # Get the best E-value for this hit
    e_value = hit.find('.//Hsp_evalue').text
    print(f"🔹 Hit: {hit_def[:50]}...")
    print(f"   E-value: {e_value}")