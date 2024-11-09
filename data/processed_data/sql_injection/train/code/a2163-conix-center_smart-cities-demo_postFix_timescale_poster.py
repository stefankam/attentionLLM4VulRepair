#!/usr/bin/python3

import psycopg2
from psycopg2 import sql

class TimescalePoster:

    def __init__(self, host, port, database, username, password):
        self.connection = psycopg2.connect(dbname=database, host=host, port=port, user=username, password=password)

    """
    Inserts data into an existing table. On failure due to not enough columns
    will automatically add columns to the table as necessary.
    Takes:
    tableName - string of the table being inserted into
    timeStamp - the timeStamp of the insertion
    tableObj - a dict of key/value pairs to insert
    """
    def insertData(self, tableName, timeStamp, tableObj):
        cols = ""
        vals = ""
        for key in tableObj:
            <fix/>cols = cols + ", {}"</fix>
            vals = vals + ", %s"

        <fix/>identifierList = []
        identifierList.append(sql.Identifier(tableName))
        for key in tableObj:
            identifierList.append(sql.Identifier(key))
</fix>
        valList = []
        <fix/>valList.append(timeStamp)</fix>
        for key in tableObj:
            valList.append(tableObj[key])


        cursor = self.connection.cursor()
        try:
            <fix/>cursor.execute(sql.SQL("INSERT INTO {} (TIMESTAMP" + cols + ") VALUES (%s" + vals + ")")
                    .format(*identifierList),valList)</fix>
            print('posted successfully!')
            self.connection.commit()
        except psycopg2.Error as e:
            cursor.close();
            <fix/>self.connection.rollback()
            print("Insert Error: {}".format(e))
            print(e.pgcode)
            #was this error due to a missing table?
            if e.pgcode == '42P01':
                print('Trying to create table')
                try:
                    self.createTable(tableName, tableObj)
                    print("Created table successfully - reinserting")
                except psycopg2.Error as e:
                    print("Failed to create table??: {}".format(e))
                    raise</fix>

                <fix/>try:
                    self.insertData(tableName, timeStamp, tableObj)
                    <fix/>print('reinserted successfully!')</fix>
                except:
                    print("Unexpected error when reinserted!")
                    raise</fix>

            <fix/>#was this error due to missing the table entirely?
            elif e.pgcode == '42703':
                columnName = e.pgerror.split('"')[1]
                print("Attempting to add column {}".format(columnName))

                identifierList = []
                typeName = None</fix>
                try:
                    <fix/>typeName = self.__getType(tableObj[columnName]);</fix>
                except TypeError as e:
                    <fix/>print("Got a type error {}".format(e))
                    print('Error with field {}'.format(columnName))</fix>
                    print('Table alteration failed')
                    raise e

                <fix/>identifierList.append(sql.Identifier(tableName));
                identifierList.append(sql.Identifier(columnName));

                query = "ALTER TABLE {} ADD COLUMN {}" + typeName
</fix>
                try:
                    cursor = self.connection.cursor()
                    <fix/><fix/>cursor.execute(sql.SQL(query).format(*identifierList))</fix></fix>
                    self.connection.commit()
                    print("Table alteration succeeded - attempting to insert again")
                except psycopg2.Error as e:
                    <fix/>self.connection.rollback()
                    print("Failed to alter table with error {}".format(e))</fix>


                try:
                    self.insertData(tableName, timeStamp, tableObj)
                    print('reinserted successfully!')
                except:
                    print("Unexpected error when reinserted!")


    """
    A private function that maps a type in python to a type in postgres
    Supports Strings, bools, numbers and arrays

    Raises a typerror on failure
    """
    def __getType(self, value):
        t = type(value)
        if(t is str):
            return 'TEXT'
        elif(t is bool):
            return 'BOOLEAN'
        elif(t is int):
            return 'DOUBLE PRECISION'
        elif(t is float):
            return 'DOUBLE PRECISION'
        elif(t is list):
            t2 = type(value[0])
            if(t2 is str):
                return 'TEXT[]'
            elif(t2 is bool):
                return 'BOOLEAN[]'
            elif(t2 is int):
                return 'DOUBLE PRECISION[]'
            elif(t2 is float):
                return 'DOUBLE PRECISION[]'
            else:
                raise TypeError('Only supports strings, booleans, numbers and arrays of the former')
        else:
            raise TypeError('Only supports strings, booleans, numbers and arrays of the former')

    """
    Creates a timescaledb table with at least a timestamp field. Partitions table
    by time.
    Takes:
    tableName - string of the table being inserted into
    tableObj - a dict of key/value pairs to start the table with
    """
    def createTable(self, tableName, tableObj):
        #find the number of columns in the in the tableobj and create
        #placeholders
        cols = ""
        for key in tableObj:
            <fix/>cols = cols + ", {{}} {}"</fix>

        <fix/>identifierList = []
        typeList = []
        identifierList.append(sql.Identifier(tableName))</fix>
        for key in tableObj:
            identifierList.append(sql.Identifier(key))
            try:
                t = self.__getType(tableObj[key])
                <fix/>typeList.append(t)</fix>
            except TypeError as e:
                <fix/>print('Error with object {} at key {} with value {}'.format(tableObj, key, tableObj[key]))
                print("Caught error {}".format(e))</fix>
                raise e

        cursor = self.connection.cursor()

        #This really really really should be safe because I'm only inserting a constrained set of types
        query = ("CREATE TABLE {{}} (TIMESTAMP TIMESTAMPTZ NOT NULL" + cols + ")").format(*typeList)
        try:
            cursor.execute(sql.SQL(query).format(*identifierList))
        except psycopg2.Error as e:
            <fix/>cursor.close()
            self.connection.rollback()
            print("CREATE TABLE Error: {}".format(e))
            raise e</fix>

        self.connection.commit()

    """
    Checks if a table exists
    takes:
    tableName - name of the table
    """
    def tableExists(self, tableName):
        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",tableName)
        except psycopg2.Error as e:
            cursor.close()
            self.connection.rollback()
            return False

        self.connection.commit()
        return True

