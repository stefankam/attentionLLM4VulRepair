#!/usr/bin/python3

import cgi
import mysql.connector
from html import beghtml, endhtml

# getting all the values from the form
form = cgi.FieldStorage()
enzyme_name   = form.getvalue('enzyme_name')
process_name  = form.getvalue('process_name')
enzyme_name2  = form.getvalue('enzyme_name2')
process_name2 = form.getvalue('process_name2')
enzyme_name3  = form.getvalue('enzyme_name3')
conc          = form.getvalue("conc")
compound      = form.getvalue("compound")
intermediate  = form.getvalue("inter")
sub           = form.getvalue("sub")
organelle     = form.getvalue("organelle")
enzyme_name3  = form.getvalue("enzyme_name3")
process_name3 = form.getvalue("process_name3")
organelle2    = form.getvalue("organelle2")
conc2         = form.getvalue("conc2")
compound2     = form.getvalue("compound2")

# establishing connection, cursor
cnx = mysql.connector.connect(user='eapfelba', database='eapfelba2', host='localhost', password='chumash1000')
query = ""
cursor = cnx.cursor()

# depending on the user input- assign the query
# if multiple text boxes are filled, the last row to be filled in will be executed
if enzyme_name:
    <fix/>query = "delete from converts where enzyme_name = %s" 
    v = (enzyme_name,)</fix>
    
<fix/>if enzyme_name3:
    query = "delete from enzyme where enzyme_name = %s"
    v = (enzyme_name3,)</fix>

if process_name:
    query = "delete from process where process_name = %s"
    v = (process_name,)
    
if process_name2 and enzyme_name2:
    <fix/>query = "delete from uses where process_name = %s and enzyme_name = %s" 
    v = (process_name2, enzyme_name2)
</fix>    
if conc and compound:
    <fix/>query = "delete from conds where concentration = %s and compound = %s"
    v = (conc, compound)
</fix>    
if intermediate:
    <fix/>query = "delete from intermediate where intermediate_name = %s"
    v = (intermediate,)
</fix>    
if organelle and sub:
    <fix/>query = "delete from location where organelle = %s and substructure = %s"
    v = (organelle, sub)</fix>

if enzyme_name3 and organelle2:
    <fix/>query = "delete from located_in where enzyme_name = %s and organelle = %s"
    v = (enzyme_name3, organelle2)
</fix>    
if process_name3 and conc2 and compound2:
    <fix/>query = "delete from operates_uner where process_name = %s and concentration = %s and compound = %s"
    v = (process_name3, conc2, compound2)</fix>

# if empty form - give the user an option to fill in something or go back to the home page
hasError = False
if not query:
    beghtml()
    print("<h3>You didn't fill anything out! :/</h3>")
    print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/delete.html">Back</a></b>')
    print('<br><b><a href = "http://ada.sterncs.net/~eapfelbaum/biobase.html">Home</a></b>')
    endhtml()
    hasError = True

# checking for errors - if there is an error, show it on the screen
try:
    <fix/>cursor.execute(query, v)</fix>
    cnx.commit()
    
except mysql.connector.Error as err:
    beghtml()
    print("Something went wrong: {}".format(err) + "<br><br>")
    print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/delete.html">Back</a></b>')
    endhtm()
    hasError = True

# otherwise print the repsonse to the screen
if hasError == False:
    # html with the response from the delete 
    beghtml()
    print("<h3>Deleted!</h3>")                                      
    print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/cgi-bin/showdb.py">Current Database</a></b><br><br>')
    print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/biobase.html">Try Something Else!</a></b><br><br>')
    print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/delete.html">Back</a></b>')
    endhtml()
    
cursor.close()
cnx.close()
