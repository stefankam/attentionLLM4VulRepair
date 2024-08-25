from parser.dfg import DFG_getter
import parser
from tree_sitter import Language, Parser

test_code = "experiment/resources/test_code.py"
code = open(test_code, 'r').read()

path = parser.dfg.__path__[0]
LANGUAGE = Language(path + "/my-languages.so", 'python')
parser = Parser()
parser.set_language(LANGUAGE)
tree = parser.parse(bytes(code, 'utf8'))
dfg, code = DFG_getter.get_data_flow(code, parser)
normalized_dfg = DFG_getter.normalize_dataflow(dfg)
print(normalized_dfg)