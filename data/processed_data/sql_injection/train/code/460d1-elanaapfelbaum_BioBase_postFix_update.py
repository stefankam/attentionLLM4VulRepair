#! /usr/bin/python3

import cgi
import mysql.connector
from html import beghtml, endhtml

# getting all the values from the html form
form = cgi.FieldStorage()
enzyme_name    = form.getvalue('enzyme_name')
product_name   = form.getvalue('product_name')
enzyme_name2   = form.getvalue('enzyme_name2')
mechanism_name = form.getvalue('mechanism_name')
process_name   = form.getvalue('process_name')
concentration  = form.getvalue('concentration')
compound_name  = form.getvalue('compound_name')
process_name2  = form.getvalue('process_name2')
goal           = form.getvalue('goal')
inter          = form.getvalue('inter')
conc           = form.getvalue('conc')
process_name3  = form.getvalue('process_name3')
enzyme_name3   = form.getvalue('enzyme_name3')
enzyme_name4   = form.getvalue('enzyme_name4')
organelle      = form.getvalue('organelle')
sub            = form.getvalue('sub')
organelle2     = form.getvalue('organelle2')
sub2           = form.getvalue('sub2')
conc2          = form.getvalue('conc2')
comp           = form.getvalue('comp')
loc            = form.getvalue('loc')
sub3           = form.getvalue('sub3')
sub4           = form.getvalue('sub4')

# establishing connection to the database
cnx = mysql.connector.connect(user='eapfelba', database='eapfelba2', host='localhost', password='chumash1000')
cursor = cnx.cursor(buffered=True)
query = ""  # initializing empty queries to avoid errors
query2 = ""

# depending on the user input assign the query
# the second query searches for the updated data in the database and shows the user what they inputed
# if multiple are filled in, the last one will be executed
if enzyme_name and product_name:
    <fix/>query = "update converts set product_name = %s where enzyme_name = %s"
    v1 = (product_name, enzyme_name)
    query2 = "select * from converts where product_name = %s and enzyme_name = %s"
    v2 = (product_name, enzyme_name)</fix>
    
if enzyme_name2 and mechanism_name:
    <fix/>query = "update enzyme set ligand_mechanism = %s where enzyme_name = %s"
    v1 = (mechanism_name, enzyme_name2)
    query2 = "select * from enzyme where enzyme_name = %s"
    v2 = (enzyme_name2,)</fix>
    
if process_name and concentration and compound_name:
    <fix/>query = "update operates_under set concentration = %s, compound = %s where process_name = %s"
    v1 = (concentration, compound_name, process_name)
    query2 = "select * from operates_under where process_name = %s"
    v2 = (process_name,)
</fix>    
if process_name2 and goal:
    <fix/>query = "update process set goal_product = %s where process_name = %s"
    v1 = (goal, process_name2)
    query2 = "select * from process where process_name = %s"
    v2 = (process_name2,)</fix>

if inter and conc:
    <fix/>query = "update intermediate set concenration = %s where intermediate_name = %s" 
    v1 = (conc, inter)
    query2 = "select * from intermediate where intermediate_name = %s" 
    v2 = (inter,)
</fix>    
if process_name3 and enzyme_name3:
    <fix/>query = "update uses set enzyme_name = %s where process_name = %s"
    v1 = (enzyme_name3, process_name3)
    query2 = "select * from uses where process_name = %s and enzyme_name = %s"
    v2 = (process_name3, enzyme_name3)
</fix>    
if enzyme_name4 and organelle and sub and sub4:
    <fix/>query = "update located_in set organelle = %s, substructure = %s where enzyme_name = %s and substructure = %s" 
    v1 = (organelle, sub, enzyme_name4, sub4)
    query2 = "select * from located_in where enzyme_name = %s"
    v2 = (enzyme_name4,)</fix>
    
if organelle2 and sub2:
    <fix/>query = "update location set substructure = %s where organelle = %s and substructure = %s"
    v1 = (sub2, organelle2, sub3)
    query2 = "select * from location where organelle = %s and substructure = %s" 
    v2 = (sub2, organelle2, sub3)
</fix>    
if conc2 and comp and loc:
    <fix/>query = "update conds set prime_location = %s where concentration = %s and compound = %s"
    v1 = (loc, conc2, comp)
    query2 = "select * from conds where concentration = %s and compound = %s and prime_location = %s"
    v2 = (conc2, comp, loc)</fix>

hasError = False
if not query: # blank form - give the user option to go back or to the home page
    beghtml()
    print("<h3>You didn't fill anything out! :/</h3>")
    print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/update.html">Back</a></b>')
    print('<br><b><a href = "http://ada.sterncs.net/~eapfelbaum/biobase.html">Home</a></b>')
    endhtml()
    hasError = True

if query: # errors
    try:
        <fix/>cursor.execute(query, v1)</fix>
        cnx.commit()

    except mysql.connector.Error as err:
        beghtml()
        print("Something went wrong: {}".format(err) + "<br><br>")
        print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/update.html">Back</a></b>')
        print('<br><b><a href = "http://ada.sterncs.net/~eapfelbaum/biobase.html">Home</a></b>')
        endhtml()
        hasError = True
        
# if there were no errors when executing the first query, continue executing the second
if hasError == False:
    <fix/>cursor.execute(query2, v2)</fix>
    data = cursor.fetchall()
    
    # html with the response from the update           
    beghtml()
    
    # if the first query did not come up with an error but the second did (typo, value not in the table)
    # i.e. the select statement came up with nothing..
    # print that something went wrong and give an option to go back
    if not data:
        print("<h3><b>Something went wrong </b></h3>")
        print("<b>Check your spelling!</b><br><br>")
        print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/update.html">Back</a></b>')
        print('<br><b><a href = "http://ada.sterncs.net/~eapfelbaum/biobase.html">Home</a></b>')
    # otherwise, you want to print out what the database not reads
    else:
        print("<h3>Updated!</h3>")
        print("The database now reads <br><br>")
        for result in data[0]:
            print("<b> | %s" % result)
        print(" | </b>")
        print("<br><br>")
        print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/cgi-bin/showdb.py">Current Database</a></b><br><br>')
        print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/biobase.html">Try Something Else!</a></b><br><br>')
        print('<b><a href = "http://ada.sterncs.net/~eapfelbaum/update.html">Back</a></b>')
    endhtml()


cursor.close()
cnx.close()
