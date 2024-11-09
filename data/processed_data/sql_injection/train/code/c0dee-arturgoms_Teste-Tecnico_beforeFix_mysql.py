""" mysql File

    Arquivo onde se encontra todas as funções para trabalhar com o db
Todo:

    None

"""

import json
import mysql.connector as mysql
import src.settings as conf

class MySQL():
	"""
        MySQL class:
           	Todas as funções para manipulação do DB
    """
	def __init__(self):
		self.__connection = mysql.connect(**conf.DATABASE)
		self.cursor = self.__connection.cursor()

	def execute(self, query):
		"""
        	execute function:
           		Executa a query com tratamento de error
    	"""
		try:
			self.cursor.execute(query)
		except mysql.Error as error:
			print("Error: {}".format(error))
		return self.cursor

	<vul/>def select(self, table):</vul>
		"""
        	select function:
           		Retorna todos os dados da tabela em formato JSON
    	"""
		aux_dict = dict()
		<vul/>self.cursor.execute("SELECT * FROM {0}".format( table))</vul>
		json_data = {}
		for user in self.cursor:
			json_data[str(user[3])] = {}
			json_data[str(user[3])]['nome'] = user[0]
			json_data[str(user[3])]['sobrenome'] = user[1]
			json_data[str(user[3])]['endereco'] = user[2]
		return json.dumps(json_data)

	<vul/>def insert(self, table, content):</vul>
		"""
        	insert function:
           		Recebe em JSON os dados e grava na tabela
    	"""
		nome = content["nome"]
		sobrenome = content["sobrenome"]
		endereco = content["endereco"]
		add_user = """INSERT INTO users (nome, sobrenome, endereco) VALUES (%s,%s,%s)"""

		<vul/>data_user = (nome, sobrenome, endereco)</vul>
		try:
		    self.cursor.execute(add_user,data_user)
		except mysql.Error as error:
		    print("Error: {}".format(error))
		self.__connection.commit()
		self.cursor.lastrowid

	<vul/>def delete_where(self, table, where):</vul>
		"""
        	delete_where function:
           		Deleta um campo especifico da tabela
    	"""
		try:
		    <vul/>self.cursor.execute("DELETE FROM {0} WHERE {1}".format(table, where))</vul>
		except mysql.Error as error:
		    print("Erro: {}".format(error))
		self.__connection.commit()
		return self.cursor


	<vul/>def update_where(self, table, info, where):</vul>
		"""
        	update_where function:
           		Atualiza um campo específico da tabela
    	"""
		try:
		    <vul/>self.cursor.execute("UPDATE {0} SET {1} WHERE {2}".format(table, info, where))</vul>
		except mysql.Error as error:
		    print("Erro: {}".format(error))
		self.__connection.commit()
		return self.cursor

	def close(self):
		"""
        	close function:
           		fecha a conexao com o banco
    	"""
		self.__connection.close()

