import asyncio
import psycopg2
import secrets

#---database table names
<fix/>connection = secrets.connection
users = "user_objects"
permissions = "rank_privileges"</fix>

<fix/>def fetch(query, parameters):
    conn = psycopg2.connect(connection)</fix>
    cur = conn.cursor()
    <fix/><fix/>cur.execute(query, parameters)</fix></fix>    
    result = cur.fetchall()
    cur.close()
    conn.close()
    return result

<fix/>def update(query, parameters):
    conn = psycopg2.connect(connection)</fix>
    cur = conn.cursor()
    cur.execute(query, parameters)
    conn.commit()
    cur.close()
    conn.close()

<fix/>def check(userid, attribute, table):
    x = fetch("SELECT {} FROM {} WHERE id  = %s;".format(attribute, table), (userid,))
    return x[0][0]

def rank_check(userid, function):
    rank = check(userid, 'rank', users)
    query = fetch("SELECT {} FROM {} WHERE FUNCTION = %s;".format(rank, permissions), (function,))
    print(rank)
    if query[0][0] == True:</fix>
        return True
    return False

<fix/>def is_int(ss):</fix>
    """ Is the given string an integer? """
    try: int(ss)
    except ValueError: return False
    else: return True

def is_valid_id(ss):
    '''verifies if id is likely a valid discord id'''
    <fix/>if type(ss) == type('') and len(ss) >= 15 and len(ss) <= 20 and is_int(ss):</fix>
        return True
    return False

