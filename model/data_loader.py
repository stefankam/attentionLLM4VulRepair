import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel
from experiment.utils import get_graph_dfg_data
from torch_geometric.data import Data

# Define dataset structure
class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, tokenizer, embedding_model, lang='python'):
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.lang = lang
        self.data = self.load_data_from_file(filepath)

    def load_data_from_file(self, filepath):
        with open(filepath, 'r') as f:
            data = []
            code_snippet = ''  # The entire code, including the vulnerable part
            fix_snippet = ''   # The entire code, including the fix part but excluding <vul> part
            vul_snippet = ''   # The vulnerable code part
            reading_vulnerability = False
            reading_fix = False
            fix_started = False  # To track if we have entered the <fix> section
            post_fix_part = False  # To track if we're after the </fix> section

            for line in f:
                if '<vul/>' in line:
                    # Start of vulnerability section
                    reading_vulnerability = True
                    code_snippet += line  # Add <vul> tag and its content to the main code snippet
                    vul_snippet = ''  # Reset vulnerability snippet
                elif '</vul>' in line:
                    # End of vulnerability section
                    reading_vulnerability = False
                    code_snippet += line  # Add </vul> tag to the main code snippet
                    vul_snippet += line.strip()  # Finalize vul_snippet
                elif '<fix/>' in line:
                    # Start of fix section
                    reading_fix = True
                    fix_started = True
                    fix_snippet += line  # Start fix_snippet here (fix section begins)
                elif '</fix>' in line:
                    # End of fix section
                    reading_fix = False
                    post_fix_part = True  # We are now after the fix section
                    fix_snippet += line.strip()  # End fix_snippet at the end of the fix section
                else:
                    # Normal lines outside of vulnerability and fix sections
                    if reading_vulnerability:
                        vul_snippet += line  # Add to vulnerability snippet
                        code_snippet += line  # Add to the main code as well
                    elif reading_fix:
                        fix_snippet += line  # Add to fix snippet only
                    else:
                        # General code outside both sections, add to both code_snippet and fix_snippet
                        code_snippet += line
                        if fix_started and post_fix_part:
                            # After </fix>, continue adding to fix_snippet
                            fix_snippet += line
                        elif not fix_started:
                            # Add to fix_snippet before <fix/>
                            fix_snippet += line

            # Append the final code, vul, and fix snippets
            if code_snippet or fix_snippet or vul_snippet:
                data.append({
                    "code": code_snippet.strip(),        # Complete code, including vulnerable part
                    "vul_snippet": vul_snippet.strip(),  # Vulnerability code section
                    "fix": fix_snippet.strip()           # Complete fix snippet, including all code except the vulnerable part
                })

        return data



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code_snippet = self.data[idx]['code']
        fix_snippet = self.data[idx]['fix']
        graph_data, _ = get_graph_dfg_data(code_snippet, self.embedding_model, self.tokenizer, lang=self.lang)
        return code_snippet, fix_snippet, graph_data,


def collate_fn(batch):
    # Ensure the structure of each item in the batch is unpacked properly
    codes = [item[0] for item in batch]  # Assuming item[0] is the code snippet as a string
    fixes = [item[1] for item in batch]  # Assuming item[1] is the fix snippet as a string
    graphs = [item[2] for item in batch]  # Assuming item[2] is graph embeddings

    # Debugging: Ensure that codes and fixes are lists of strings
    assert isinstance(codes, list) and all(isinstance(code, str) for code in codes), "Codes must be a list of strings"
    assert isinstance(fixes, list) and all(isinstance(fix, str) for fix in fixes), "Fixes must be a list of strings"

    # Continue with tokenization and further processing
    tokenized_codes = tokenizer(codes, return_tensors='pt', truncation=True, padding=True, max_length=512)
    tokenized_fixes = tokenizer(fixes, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Generate embeddings and process graph embeddings
    with torch.no_grad():
        code_outputs = embedding_model(**tokenized_codes)
        fix_outputs = embedding_model(**tokenized_fixes)

    sequence_embeddings = code_outputs.last_hidden_state
    fix_embeddings = fix_outputs.last_hidden_state

    # Determine max size for padding
    max_graph_size = max(graph.x.size(0) for graph in graphs)  # Assuming x is the node feature tensor

    # Pad graph embeddings
    padded_graphs = []
    for graph in graphs:
        num_nodes = graph.x.size(0)
        if num_nodes < max_graph_size:
            padding_size = max_graph_size - num_nodes
            # Pad node features
            padded_x = F.pad(graph.x, (0, 0, 0, padding_size), mode='constant', value=0)
            # Pad edge_index
            padded_edge_index = torch.cat([graph.edge_index, torch.zeros(2, padding_size).long()], dim=1)
            padded_graphs.append(Data(x=padded_x, edge_index=padded_edge_index))
        else:
            padded_graphs.append(graph)


    return codes, fixes, padded_graphs, sequence_embeddings, fix_embeddings


# Initialize tokenizer and embedding model
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
embedding_model = RobertaModel.from_pretrained("Salesforce/codet5-base")

# File containing code snippets with vulnerability tags and corresponding labels
code_file = '/content/graphLLM4VulRepair/experiment/resources/test_code.py'

# Create the dataset and DataLoader
dataset = CodeDataset(code_file, tokenizer, embedding_model)
data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
