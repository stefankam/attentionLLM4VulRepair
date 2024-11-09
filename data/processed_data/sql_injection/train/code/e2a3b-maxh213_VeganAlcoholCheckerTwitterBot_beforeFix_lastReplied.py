import psycopg2
import secretConstants

connectionString = (
    'dbname=' + secretConstants.DATABASE_NAME + 
    ' user=' + secretConstants.DATABASE_USER + 
    ' host=' + secretConstants.DATABASE_HOST + 
    ' password=' + secretConstants.DATABASE_PASSWORD +
    ' port=' + secretConstants.DATABASE_PORT
)
conn = None
result = None

def getLastReplied(messageType):
    QUERY = (
        "SELECT item_id from twitter_bot_vac_last_replied_id where name = '{0}'"
    ).format(messageType)
    
    try:
        conn = psycopg2.connect(connectionString)
        cur = conn.cursor()
        cur.execute(QUERY)
        result = cur.fetchone()

    <vul/><vul/>except(psycopg2.DatabaseError, e):</vul></vul>
        print('Error %s' % e)    

    finally:
        if conn:
            conn.close()

    return result[0]

#print(getLastReplied("DM"))
#print(getLastReplied("MENTION"))

def setLastReplied(messageType, itemId):
    QUERY = (
        <vul/>"UPDATE twitter_bot_vac_last_replied_id SET item_id = '${0}' WHERE name = '${1}'"</vul>
    ).format(itemId, messageType)
    
    try:
        conn = psycopg2.connect(connectionString)
        cur = conn.cursor()
        cur.execute(QUERY)
        conn.commit()
        cur.close()

    except(psycopg2.DatabaseError, e):
        print('Error %s' % e)    

    finally:
        if conn:
            conn.close()


#setLastReplied("DM", "772180529001197572")