import os.path
import shutil

from data.process.utils import read_patches, download_vulnerable_files, get_github_client


vulnerability = "sql_injection"
token = ""
train_and_valid_ratio = 0.8
numb_patches = 1
isTrain = False

data_file = "data/raw_data/{}.json".format(vulnerability)
output_path = "data/processed_data/" + vulnerability
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

g = get_github_client(token)

records = read_patches(data_file)
total_patches = len(records)
if isTrain:
    test_start_point = 0
else:
    test_start_point = int(total_patches * train_and_valid_ratio)

graped_records = records[test_start_point:test_start_point + numb_patches]

num_available_commits = download_vulnerable_files(graped_records, output_path, g)

print("Number of available records: {}".format(num_available_commits))




