import pymysql
import dbconfig

class DBHelper:

	def connect(self, datbase="crimemap"):
		return pymysql.connect(host='localhost',
							user=dbconfig.db_user,
							passwd=dbconfig.db_password,
							db=datbase)

	def get_all_inputs(self):
	<vul/>connection = self.connect()</vul>
		try:
			query = "SELECT description FROM crimes;"
			with connection.cursor() as cursor:
				<vul/>cursor.execute(query)</vul>
			return cursor.fetchall()
		finally:
			connection.close()

	def add_input(self, data):
		connection = self.connect()
		try:
			<vul/>query = "INSERT INTO crimes (description) VALUES ('{}');".format(data) #i didn't understand this '.format(data)'</vul>
			with connection.cursor() as cursor:
				cursor.execute(query)
				connection.commit()
		finally:
			connection.close()

	def clear_all(self):
		<vul/>connection.connect(self):</vul>
		try:
			query = "DELETE FROM crimes;"
			with connection.cursor() as cursor:
				cursor.execute(query)
				connection.commit()
		finally:
			connection.close()
