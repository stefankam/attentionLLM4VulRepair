import sqlite3

class Sqlite():
    __db : sqlite3.Connection

    def __init__(self, file: str) -> None:
        """This method is designed to initialize the Sqlite class.

        Args:
            file (str): The path to the database file.
        """
        self.__db =  sqlite3.connect(file)

    <fix/>def select(self, query:str, parameters: list = []) -> dict:</fix>
        """This method is designed to execute a SQL SELECT query.

        Args:
            query (str): The SQL query
            parameters (list): The parameters to pass to the query (default: [])

        Returns:
            dict: A dict with the informations who were fetch on the database.
        """
        cursor = self.__db.cursor()
        <fix/><fix/>cursor.execute(query, parameters)</fix></fix>

        result = cursor.fetchall()

        return result

    <fix/>def modify(self, query:str, parameters: list = []) -> None:</fix>
        """This method is designed to execute a SQL query (Insert or Update).

        Args:
            query (str): The SQL query
            parameters (list): The parameters to pass to the query (default: [])
        """
        cursor = self.__db.cursor()
        cursor.execute(query, parameters)

        self.__db.commit()

