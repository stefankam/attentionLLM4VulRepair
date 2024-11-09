import os.path

from data.process.utils import collect_codes_single_data

raw_data_dir = "data/raw_data"
output_dir = "data/processed_data"
token = "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"

max_vulnerability = 8

num_vulnerability = 0
for vulnerability in os.listdir(raw_data_dir):
    # Skip hidden files (those starting with a dot) or non-JSON files
    if vulnerability.startswith('.') or not vulnerability.endswith('.json'):
        continue
    print(f"Processing {vulnerability}")
    vulnerability = vulnerability.replace(".json", "")
    if num_vulnerability == max_vulnerability:
        break
    for data_type in ["train", "valid", "test"]:
        collect_codes_single_data(vulnerability, token, data_type, raw_data_dir, output_dir, add_tag=False,
                                  numb_patches=-1)
    num_vulnerability += 1
print("Done!")
