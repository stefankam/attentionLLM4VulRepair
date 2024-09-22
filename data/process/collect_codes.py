import os.path

from data.process.utils import collect_codes_single_data

raw_data_dir = "data/raw_data"
output_dir = "data/processed_data"
token = ""

for vulnerability in os.listdir(raw_data_dir):
    print(f"Processing {vulnerability}")
    vulnerability = vulnerability.replace(".json", "")
    for data_type in ["train", "valid", "test"]:
        collect_codes_single_data(vulnerability, token, data_type, raw_data_dir, output_dir, numb_patches=1)

print("Done!")