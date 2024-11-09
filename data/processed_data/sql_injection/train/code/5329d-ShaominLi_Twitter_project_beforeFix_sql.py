import psycopg2



#链接数据库
def connectDB(dbname,uname,psw):
    #conn=psycopg2.connect(database="test",user="lishaomin",password="19931004",host="127.0.0.1",port="5432")
    conn=psycopg2.connect(database=dbname,user=uname,password=psw,host="127.0.0.1",port="5432")
    return conn


#查询数据库
def queryDB(conn,sql_select):
    print("query data")
    cur=conn.cursor()
    <vul/>#sql_select="select * from users;"
    cur.execute(sql_select)
    rows=cur.fetchall()
    #for row in rows:
    #print ("user:%s"%(row[1]))
    return rows</vul>



#插入数据
<vul/>def insertDB(conn,sql_insert):</vul>
    cur=conn.cursor()
    <vul/>result=cur.execute(sql_insert)
    conn.commit()
    print("insert data successfull")
    return result</vul>

#delete data
<vul/>def deleteDB(conn,sql_delete):</vul>
    cur=conn.cursor()
    <vul/>result=cur.execute(sql_delete)
    conn.commit()
    print("delete data successfull")
    return result</vul>


#update data
<vul/>def updateDB(conn,sql_update):</vul>
    cur=conn.cursor()
    <vul/>result=cur.execute(sql_update)
    conn.commit()
    print("update data successfull")
    return result</vul>


#关闭链接
def closeDB(conn):
    conn.close()



