import ast


# Generate the AST nodes for a given code snippet
def get_ast_nodes(code_snippet):
    return ast.walk(ast.parse(code_snippet))


code = "def foo():\n    print('Hello, World!')\n"
print(get_ast_nodes(code))
