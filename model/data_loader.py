

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from experiment.utils import get_graph_dfg_data
from torch_geometric.data import Data

import os

# Define dataset structure
class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, tokenizer, embedding_model, lang='python'):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.lang = lang
        self.data = self.load_data_from_directory(filepath)

    def load_data_from_directory(self, filepath):
        data = []

        # Get a list of all files in the directory and sort them
        files = sorted(os.listdir(filepath))

        # Filter the files to separate beforeFix and postFix
        before_fix_files = [f for f in files if 'beforeFix' in f]
        post_fix_files = [f for f in files if 'postFix' in f]
        # Debugging: Print the found files
        print(f"BeforeFix files: {before_fix_files}")
        print(f"PostFix files: {post_fix_files}")

        # Ensure there's a corresponding PostFix file for each BeforeFix file
        for before_file, post_file in zip(before_fix_files, post_fix_files):
            # Read the BeforeFix file (code_snippet)
            with open(os.path.join(filepath, before_file), 'r') as before_fix_file:
                before_fix_content = before_fix_file.read()

            # Read the PostFix file (fix_snippet)
            with open(os.path.join(filepath, post_file), 'r') as post_fix_file:
                post_fix_content = post_fix_file.read()

            # Append to the data list
            data.append({
                "code": before_fix_content.strip(),  # The code before the fix (code_snippet)
                "fix": post_fix_content.strip()      # The code after the fix (fix_snippet)
            })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code_snippet = self.data[idx]['code']
        fix_snippet = self.data[idx]['fix']
        graph_result = get_graph_dfg_data(code_snippet, self.embedding_model, self.tokenizer, lang=self.lang)
        if graph_result is None:
            print("Warning: No graph data generated for this sample.")
            return None  # Skip this sample in the dataset if no graph data
        graph_data, _ = graph_result

        return code_snippet, fix_snippet, graph_data,

def collate_fn(batch, tokenizer, embedding_model, max_length, device):
    # Filter out None values from the batch list
    batch = [item for item in batch if item is not None and len(item) == 3]
 
    if not batch:  # Handle cases where the entire batch is filtered out
        return None

    # Ensure the structure of each item in the batch is unpacked properly
    codes = [item[0] for item in batch]  # Assuming item[0] is the code snippet as a string
    fixes = [item[1] for item in batch]  # Assuming item[1] is the fix snippet as a string
    graphs = [item[2] for item in batch]  # Assuming item[2] is graph embeddings

    # Debugging: Ensure that codes and fixes are lists of strings
    assert isinstance(codes, list) and all(isinstance(code, str) for code in codes), "Codes must be a list of strings"
    assert isinstance(fixes, list) and all(isinstance(fix, str) for fix in fixes), "Fixes must be a list of strings"

    # Tokenize code and fix snippets
    tokenized_codes = tokenizer(codes, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    tokenized_fixes = tokenizer(fixes, return_tensors='pt', truncation=True, padding=True, max_length=max_length)

    # Move tokenized tensors to the correct device
    tokenized_codes = {key: value.to(device) for key, value in tokenized_codes.items()}
    tokenized_fixes = {key: value.to(device) for key, value in tokenized_fixes.items()}

    code_token_ids = tokenized_codes['input_ids'].to(device)
    fix_token_ids = tokenized_fixes['input_ids'].to(device)

    # Ensure the shapes of tokenized codes and fixes are consistent
#    assert code_token_ids.size(1) == fix_token_ids.size(1), f"Code and fix token sizes don't match! {code_token_ids.size(1)} vs {fix_token_ids.size(1)}"

    # Run the embedding model
    with torch.no_grad():
        code_outputs = embedding_model(**tokenized_codes)
        fix_outputs = embedding_model(**tokenized_fixes)

    sequence_embeddings = code_outputs.last_hidden_state.to(device)
    fix_embeddings = fix_outputs.last_hidden_state.to(device)

    # Handle graph data
    max_graph_size = max(graph.x.size(0) for graph in graphs)
    padded_graphs = []
    for graph in graphs:
        if graph.x.device != device:
            print(f"Graph data not on {device}, moving now.")
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)

        num_nodes = graph.x.size(0)
        if num_nodes < max_graph_size:
            padding_size = max_graph_size - num_nodes
            padded_x = F.pad(graph.x, (0, 0, 0, padding_size), mode='constant', value=0)
            padded_edge_index = torch.cat([graph.edge_index, torch.zeros(2, padding_size).long().to(device)], dim=1)
            padded_graphs.append(Data(x=padded_x.to(device), edge_index=padded_edge_index))
        else:
            padded_graphs.append(graph)


    print(f"Embedding Model Device: {next(embedding_model.parameters()).device}")
    print(f"Input IDs Device: {tokenized_codes['input_ids'].device}")
    print(f"Attention Mask Device: {tokenized_codes['attention_mask'].device}")
    return code_token_ids, fix_token_ids, codes, fixes, padded_graphs, sequence_embeddings, fix_embeddings

def get_dataload(device, max_length, batch_size=2, vulnerability='command_injection', loader_type='train'):
    config = RobertaConfig.from_pretrained("Salesforce/codet5-base")
    config.max_position_embeddings = max_length  # Increase max position embeddings
    embedding_model = RobertaModel.from_pretrained("Salesforce/codet5-base", config=config).to(device)
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", config=config)

    # File containing code snippets with vulnerability tags and corresponding labels
    filepath = ("/home/skb67/attentionLLM4VulRepair/data/processed_data/{}/{}/code".format(vulnerability, loader_type))

    # Create the dataset and DataLoader
    dataset = CodeDataset(filepath, tokenizer, embedding_model)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             collate_fn=lambda b: collate_fn(b, tokenizer, embedding_model, max_length),
                             shuffle=True, 
                             generator=torch.Generator(device=device))

    return data_loader
