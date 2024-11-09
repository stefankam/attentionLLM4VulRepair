import sqlite3

class Sqlite():
    __db : sqlite3.Connection

    def __init__(self, file: str) -> None:
        """This method is designed to initialize the Sqlite class.

        Args:
            file (str): The path to the database file.
        """
        self.__db =  sqlite3.connect(file)

    <vul/>def select(self, query:str) -> dict:</vul>
        """This method is designed to execute a SQL SELECT query.

        Args:
            query (str): The SQL query

        Returns:
            dict: A dict with the informations who were fetch on the database.
        """

        <vul/><vul/>cursor = self.__db.cursor()
        cursor.execute(query)</vul></vul>

        result = cursor.fetchall()

        return result

    <vul/>def modify(self, query:str) -> None:</vul>
        """This method is designed to execute a SQL query (Insert or Update).

        Args:
            query (str): The SQL query
        """

        cursor = self.__db.cursor()
        cursor.execute(query)

        self.__db.commit()

