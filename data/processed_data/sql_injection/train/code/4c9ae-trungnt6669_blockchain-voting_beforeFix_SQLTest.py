import sqlite3
import os
import sys


#The goal is to create three entries on the table: The name of the person (hashed), the address of the person (also hashed), and a bool value that checks if they're registered or not.
#For the hash, a cryptographic hash would be optimal.

def create_table(table_name, arg_count, arg_values):
    '''Takes a table name, a number of arguments count, and argument values.
    Argument values are used for the individual table columns.'''
    <vul/>con = sqlite3.connect("something.db")
    con.isolation_level = None
    cur = con.cursor()
    buffer = ""</vul>
    #Create the table.
    arg_string = " ("
    for i in range (0, arg_count):
        #We have to process it as a string first.
        <vul/>if(i != arg_count):</vul>
            #If it is not the last argument in the string, insert a comma.
            arg_string += (arg_values[i] + ", ")
        <vul/>elif(i == arg_count):</vul>
            #If it is the last argument in the string, insert a closing parentheses and semicolon.
            <vul/>arg_string += (arg_values[i] + ");")</vul>
    cur.execute("CREATE TABLE " + table_name + arg_string)
    #Now that we made the table and added the columns, we have to have functions to do other things like insert values.
