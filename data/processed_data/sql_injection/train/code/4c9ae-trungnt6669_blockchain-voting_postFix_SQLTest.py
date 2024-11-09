<fix/></fix>import sqlite3
import os
import sys


#The goal is to create three entries on the table: The name of the person (hashed), the address of the person (also hashed), and a bool value that checks if they're registered or not.
#For the hash, a cryptographic hash would be optimal.

#We have to open the server connection first.
con = sqlite3.connect("something.db")
con.isolation_level = None
cur = con.cursor()
buffer = ""

def create_table(table_name, arg_count, arg_values):
    '''Takes a table name, a number of arguments count, and argument values.
    Argument values are used for the individual table columns.'''

    #Create the table.
    arg_string = " ("
    for i in range (0, arg_count):
        #We have to process it as a string first.
        <fix/>if i != (arg_count - 1):</fix>
            #If it is not the last argument in the string, insert a comma.
            arg_string += (arg_values[i] + ", ")
        <fix/>elif i == (arg_count - 1):</fix>
            #If it is the last argument in the string, insert a closing parentheses and semicolon.
            <fix/>arg_string += (arg_values[i] + ") ")</fix>
    cur.execute("CREATE TABLE " + table_name + arg_string)
    con.commit() #Commit changes
    #Now that we made the table and added the columns, we have to have functions to do other things like insert values.

def insert_entries(table_name, table_col_count, col_values):
    #We insert values in an entire block for a chosen table. Insert it as name of table,  a numerical length of list, and the list itself.
    #Insert values as a list. A format example is included.
    script_char = 'INSERT INTO ' + table_name + ' VALUES ('
    # We have to use ? and other methods to sanitize inputs. There is an else clause to reject the input.
    for i in range (table_col_count):
        if i < (table_col_count - 1):
            script_char += '?,'
        elif i == (table_col_count - 1):
            script_char += '?)'
        else:
            #Print error message.
            print 'Bad Input. Rejecting...'
            #Reject input.
            return
    cur.executemany(script_char, col_values)
