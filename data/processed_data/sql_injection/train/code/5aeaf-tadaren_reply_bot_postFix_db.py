import psycopg2

config = {}

def get_all():
    connection = psycopg2.connect(host=config['HOST'], port=config['PORT'], database=config['NAME'], user=config['USER'], password=config['PASSWORD'])
    cur = connection.cursor()
    cur.execute("select * from reply_map")
    out = {}
    for row in cur:
        out[row[0]] = row[1]
    connection.commit()
    return out

def insert(key, value):
    connection = psycopg2.connect(host=config['HOST'], port=config['PORT'], database=config['NAME'], user=config['USER'], password=config['PASSWORD'])
    cur = connection.cursor()
    try:
        <fix/>cur.execute("insert into reply_map values(?, ?)", (key, value))</fix>
        connection.commit()
    except:
        pass

if __name__ == '__main__':
    connection = psycopg2.connect(host=config['HOST'], port=config['PORT'], database=config['NAME'], user=config['USER'], password=config['PASSWORD'])
    cur = connection.cursor()
    cur.execute("create table reply_map(key text not null unique, value text not null)")
    connection.commit()
