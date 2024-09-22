import os.path

from data.process.utils import collect_codes_single_data

raw_data_dir = "data/raw_data"
output_dir = "data/processed_data"
token = "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"

for vulnerability in os.listdir(raw_data_dir):
    print(f"Processing {vulnerability}")
    vulnerability = vulnerability.replace(".json", "")
    for data_type in ["train", "valid", "test"]:
        collect_codes_single_data(vulnerability, token, data_type, raw_data_dir, output_dir, numb_patches=1)

print("Done!")