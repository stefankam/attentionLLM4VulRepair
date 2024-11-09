<fix/></fix>import psycopg2
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

def getAlcoholByName(name):
    name = fixTypingErrors(name)
    name = "%" + name + "%"
    QUERY = (
        "SELECT barnivore_product_name, barnivore_status, barnivore_country " + 
        "FROM barnivore_product " +
        <fix/>"WHERE lower(barnivore_product_name) like lower(%s)"</fix>
    )
        
    try:
        conn = psycopg2.connect(connectionString)
        cur = conn.cursor()
        <fix/>cur.execute(QUERY, (name,))</fix>
        result = cur.fetchall()

    <fix/>except psycopg2.DatabaseError as e:</fix>
        print('Error %s' % e)    

    finally:
        if conn:
            conn.close()

    return result
    

def fixTypingErrors(name):
    name = name.lower() 
    if name == "guiness":
        name = "guinness"
    return name

#Uncomment for testing
#print(getAlcoholByName("Budweiser"))


