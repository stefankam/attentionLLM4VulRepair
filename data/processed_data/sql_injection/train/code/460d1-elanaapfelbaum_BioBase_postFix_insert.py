#!/usr/bin/python3                                     
                                
import cgi
import mysql.connector
from html import beghtml, endhtml

# getting the values from the html form
form = cgi.FieldStorage()
insert_table = form.getvalue('insert_table')
values       = form.getvalue('values')

<fix/># if values were inserted, split on the comma and concatenate the right amount of %s for the prepared statement</fix>
if values:
    values = values.split(', ')
    valueQuery = "("
    for value in values:
        <fix/>valueQuery += "%s, "
    valueQuery = valueQuery[:-2] + ")"</fix>

# mysql connection
cnx = mysql.connector.connect(user='eapfelba', host = 'localhost', database='eapfelba2', password='chumash1000')
query=""  # intialized as empty to prevent errors
cursor = cnx.cursor()

# creating the query based on the user input
# the insert_table cannot be inserted using prepared statements bc of the implicitly assigned quotes- this is vulnerable to SQL injection (even though only from the drop down)
if insert_table and values:
    <fix/>query = "insert into " + insert_table + " values " + valueQuery    
    v = tuple(values)  # making a tuple of the values inputed from the form to put in the execute statement
</fix>    
# checking for errors
hasError = False
if not query:  # empty form
    beghtml()
    print("<h3>You didn't fill anything out! :/</h3>")
    print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/insert.html">Back</a></b>')
    print('<br><b><a href = "http://ada.sterncs.net/~eapfelbaum/biobase.html">Home</a></b>')
    endhtml()
    hasError = True
    
if query:
    try: # try to execute the query, otherwise print out the issue on an html page and give the user options to go back
        <fix/>cursor.execute(query, v)</fix>
        cnx.commit()   
    except mysql.connector.Error as err:
        beghtml()
        print("Something went wrong: {}".format(err) + "<br><br>")
        print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/insert.html">Back</a></b>')
        endhtml()
        hasError = True

# if there is no error, print out the results!
if hasError == False:
    beghtml()
    print("<h3>")
    # print them out in the right format for the results page
    <fix/>for s in values:
        print("<b> | %s" % s)</fix>
    print(" | </b></h3>")
    print("<h3>is now in the table %s!</h3>" % insert_table)
    print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/cgi-bin/showdb.py">Current Database</a></b><br><br>')
    print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/biobase.html">Try Something Else!</a></b><br><br>')
    print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/insert.html">Back</a></b>')
    endhtml()
    
cursor.close()
cnx.close()
