#! /usr/bin/env python3
#This is the orderFilledScript that updates the DB when you laser keys.

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

openConn = False

#Determines wether the user has confirmed the information was typed correctly
confirmed = None

#Determines wether the code should loop for more key entries
orderComplete = False

#Marks that it's not the first loop, so don't ask for order number again
multiKeyOrder = False

#Used to navigate loop where user decides to loop code or not
addMore = None

u_date = datetime.now()

while orderComplete == False:
    try:
        #Taking order info from user
        clear()
        while confirmed != "yes":
            print('Update the key inventory by entering the order information below.')
            divider()
            if multiKeyOrder == False:
                u_orderNum = input('Order #: ')
            u_keyNum = input('Key used (i.e. #29): #')
            u_keysUsed = input('# of keys lased: ')
            clear()
            #Display info and have user confirm if it's correct before committing
            divider()
            print( "{} \n Order #: {} \n Key #: {} \n # of keys lased: {}".format(
                u_date, u_orderNum, u_keyNum, u_keysUsed))
            divider()
            confirmed = input("Is the information above correct? ")
            #If yes then insert this information into the database
            if confirmed == "yes":
                clear()
            #If no prompt them to re-enter the information properly
            elif confirmed == "no":
                clear()
                print('Re-enter the information. \n')
            #If they enter anything other than yes or no, ask again
            else:
                clear()
                print("Must answer yes or no, it's case sensitive because I'm lazy! \n")
        #Reset confirmed for future while loops
        confirmed = "No"
        #Convert some of the user input values to int
        u_keyNum = int(u_keyNum)
        u_keysUsed = int(u_keysUsed)
        #connect to db
        print('Connecting to database...')
        db = pyodbc.connect(Driver='{SQL Server Native Client 11.0}',
                    Server='(LocalDB)\\LocalDB Laser',
                    Database='laserInv',
                    trusted_connection='yes')
        openConn = True
        #cursor 1 to get preCount value
        c1 = db.cursor()
        <vul/>c1.execute("SELECT invCount FROM keyInventory WHERE keyNum = '%s';" % u_keyNum)</vul>
        try: 
            #Check to see if cursor one has A result.
            u_preCount = (c1.fetchall()[0][0])
        except IndexError:
            divider()
            print("ERROR: The key number you entered doesn't exist in the keyInventory table.")
            print("TIP: If you know you've typed it correctly, you'll have to add it to the Database with newKey.py") 
            divider()
            input("Press Enter to close...")
            if openConn == True:
                db.close()
                openConn = False
            sys.exit()
        except Exception:
            if openConn == True:
                db.close()
                openConn = False
                divider()
                raise
                divider()
                input("Press Enter to close...")
            sys.exit()
        #Grab datetime for this commit
        u_date = datetime.now()
        #Calculate postCount
        u_postCount = u_preCount - u_keysUsed
        #Insert all the information into ordersFilled Table
        c2 = db.cursor()
        <vul/>c2.execute("INSERT INTO ordersFilled (submit_time, orderNum, keyNum, keysUsed, preCount, postCount) VALUES (?, ?, ?, ?, ?, ?);", (u_date, u_orderNum,  u_keyNum, u_keysUsed, u_preCount, u_postCount))</vul>
        c2.commit()
        #Insert the new inventory count into keyInv Table
        c3 = db.cursor()
        <vul/>c3.execute("UPDATE keyInventory SET invCount = ? WHERE keyNum = ?;", (u_postCount, u_keyNum))</vul>
        c3.commit()
        clear()
        print('Success! Database has been updated.')
        addMore = None
        #While loop to ask user if they want to remove more keys from inventory
        while addMore != "yes" and addMore != "no": 
            addMore = input('Are there more keys on this order? ')
            if addMore == 'yes':
                #do nothing
                multiKeyOrder = True
                if openConn == True:
                    db.close()
                    openConn = False
            elif addMore == 'no':
                orderComplete = True
                print('Okay, bye!')
            else:
                clear()
                print("Must answer yes or no, it's case sensitive because I'm lazy!")
    except Exception:
    	raise
    finally:
        if openConn == True:
            db.close()
            openConn = False
sys.exit()