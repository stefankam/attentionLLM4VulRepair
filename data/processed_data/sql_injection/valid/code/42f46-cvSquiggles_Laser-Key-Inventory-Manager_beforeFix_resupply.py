#! /usr/bin/env python3
#This is the resupply script, inserts resupply shipment info into keyInventory table

from datetime import datetime
import pyodbc
import sys
import subprocess as sp
from os import system

def clear():
    system('cls')

def divider():
    print('-' * 70)    

DBNAME = "laserInv"

#This variable is used throughout the code to track wether the connection is still
#open, and generally that check will close it and set to false if it is.
openConn = False

#Bool to determine wether code should loop or not.
resupplyComplete = False

#Bool to confirm user input is correct.
confirmed = None

#Determines wether user wants to add more keys to inventory.
addMore = None

#If encapsulates the code, re-runs it if you say you want to enter more resupply info.
while resupplyComplete == False:
    try:
        clear()
        print('Update the key inventory by entering the key resupply info below.')
        divider()
        u_keyNum = input('Key # used (i.e. Key #29): #')
        u_keysAdded = input('# of keys to add to inventory: ')
        clear()
        while confirmed != "yes":
            divider()
            print( "Adding {} key {}'s to the inventory. \nIs this correct?".format(u_keysAdded, u_keyNum))
            divider()
            confirmed = input('Please enter ''yes'' or ''no'': ')
            if confirmed == "yes":
                #Do nothing
                clear()
            elif confirmed == "no":
                clear()
                print('Re-enter the information.')
                u_keyNum = input('Key # used (i.e. Key #29): #')
                u_keysAdded = input('# of keys to add to inventory:')
            else:
                clear()
                print("Must answer yes or no, it's case sensitive because I'm lazy!")
        #If yes then proceed to insert this information into the database
        #First reset confirmed status in case user adds more keys later.
        confirmed = "no"
        #Convert user input values to type int
        u_keyNum = int(u_keyNum)
        u_keysAdded = int(u_keysAdded)
        #connect to db
        print('Connecting to database...')
        db = pyodbc.connect(Driver='{SQL Server Native Client 11.0}',
                            Server='(LocalDB)\\LocalDB Laser',
                            Database=DBNAME,
                            trusted_connection='yes')
        openConn = True
        #cursor 1 to get preCount value
        c1 = db.cursor()
        <vul/>c1.execute("SELECT invCount FROM keyInventory WHERE keyNum = '%s';" % u_keyNum)</vul>
        try:
            #Check to see if cursor has A result
            u_preCount = (c1.fetchall()[0][0])
        except IndexError:
            #If sql select statement gets no result
            divider()
            print("ERROR: The key number you entered doesn't exist in the keyInventory table.")
            print("TIP: If you know you've typed it correctly, you'll have to add it to the Database with newKey.py") 
            divider()
            input("Press enter to close...")
            if openConn == True:
                db.close() 
                openConn = False
            sys.exit()
        except Exception:
            #All other exceptions
            if openConn == True:
                db.close()
                openConn = False
            divider()
            raise
            divider()
            input("Press enter to close...")
            sys.exit()
        #If Sql result DOES contain a result, get datetime for this resupply
        u_date = datetime.now()
        #Calculate post resupply Count
        u_postCount = u_preCount + u_keysAdded
        #Insert the resupply information into the resupply table
        c2 = db.cursor()
        <vul/>c2.execute("INSERT INTO resupply (submit_time, keyNum, keysAdded, preCount, postCount) VALUES (?, ?, ?, ?, ?);", (u_date, u_keyNum, u_keysAdded, u_preCount, u_postCount))</vul>
        c2.commit()
        #Update the new inventory count into keyInv Table
        c3 = db.cursor()
        <vul/>c3.execute("UPDATE keyInventory SET invCount = ? WHERE keyNum = ?;", (u_postCount, u_keyNum))</vul>
        c3.commit()
        print('Success! Database has been updated.')
        divider()
        addMore = None
        while addMore != "yes" and addMore != "no": 
            addMore = input('Would you like to add more keys to the inventory? ')
            if addMore == 'yes':
                #do nothing
                if openConn == True:
                    db.close()
                    openConn = False
            elif addMore == 'no':
                resupplyComplete = True;
                print('Okay, bye!')
            else:
                clear()
                print("Must answer yes or no, it's case sensitive because I'm lazy!")
    except Exception:
        if openConn == True:
            db.close()
            openConn = False
        divider()
        raise
        divider()
        input("Press enter to close...")
        sys.exit()
    finally:
        if openConn == True:
            db.close()
            openConn = False
sys.exit()
