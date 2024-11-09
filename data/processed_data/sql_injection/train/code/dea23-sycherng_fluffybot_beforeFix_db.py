import psycopg2

#---database table names
<vul/>user_table = 'user_objects'
rank_permit_table = 'rank_privileges'</vul>

<vul/>def fetch(query):
    conn = psycopg2.connect("dbname=fluffy_bot user=censored password=Laumau11p")</vul>
    cur = conn.cursor()
    <vul/><vul/>cur.execute(query)</vul></vul>
    result = cur.fetchall()
    cur.close()
    conn.close()
    return result

<vul/>def update(query):
    conn = psycopg2.connect("dbname=fluffy_bot user=censored password=Laumau11p")</vul>
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    cur.close()
    conn.close()

<vul/>def check(id, attribute, clss):
    try:
        attribute_value = fetch("SELECT {} FROM {} WHERE id = '{}';".format(attribute, clss, id))
    except:
        return None
    else:
        return attribute_value

def rank_check(id, function):
    query = check(id, 'rank', user_table)
    rank = query[0][0]
    query2 = fetch("SELECT {} FROM {} WHERE FUNCTION = '{}';".format(rank, rank_permit_table, function))
    if query2[0][0] == True:</vul>
        return True
    return False

<vul/>def isInt(ss):</vul>
    """ Is the given string an integer? """
    try: int(ss)
    except ValueError: return False
    else: return True

def is_valid_id(ss):
    '''verifies if id is likely a valid discord id'''
    <vul/>if type(ss) == type('') and len(ss) >= 15 and len(ss) <= 20 and isInt(ss):</vul>
        return True
    return False

