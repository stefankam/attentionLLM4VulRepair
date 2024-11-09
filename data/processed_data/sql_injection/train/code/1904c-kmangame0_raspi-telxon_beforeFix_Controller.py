import sqlite3 as sql

class DB_Connector:

	<vul/>def __init__(self, *args, **kwargs):</vul>

		<vul/>self.dbStr = "../Databases/Product.db"</vul>
		self.table_name = 'products'
		self.id_column  = 'id'
		self.column_2   = 'upc'
		self.column_3   = 'name'
		self.column_4   = 'image'
		self.some_upc = ""
		self.result = ""

	
	def fetch_product(self):

		conn = sql.connect(self.dbStr)

		c = conn.cursor()

		<vul/>c.execute("SELECT * FROM {tn} WHERE {upc}={my_upc}".\
	        format(tn=self.table_name, cn=self.column_2, 
	        	upc=self.column_2, my_upc=self.some_upc))</vul>
		
		result = c.fetchone()

		return result