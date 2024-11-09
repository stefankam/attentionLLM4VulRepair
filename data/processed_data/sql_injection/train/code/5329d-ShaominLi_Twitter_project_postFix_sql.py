import psycopg2



#链接数据库
def connectDB(dbname,uname,psw):
    conn=psycopg2.connect(database=dbname,user=uname,password=psw,host="127.0.0.1",port="5432")
    return conn


#查询数据库
<fix/>def queryDB(conn,sql_select,params):</fix>
    print("query data")
    cur=conn.cursor()
    <fix/>try:
        cur.execute(sql_select,params)
        rows=cur.fetchall()
    except Exception as err:
        closeDB(conn)
        print(err)
    else:
        return rows</fix>



#插入数据
<fix/>def insertDB(conn,sql_insert,params):</fix>
    cur=conn.cursor()
    <fix/>try:
        cur.execute(sql_insert,params)
        conn.commit()
    except Exception as err:
        closeDB(conn)
        print(err)
    else: 
        print("insert data successfull")</fix>

#delete data
<fix/>def deleteDB(conn,sql_delete,params):</fix>
    cur=conn.cursor()
    <fix/>try:
        cur.execute(sql_delete,params)
        conn.commit()
    except Exception as err:
        closeDB(conn)
        print(err)
    else: 
        print("delete data successfull")</fix>


#update data
<fix/>def updateDB(conn,sql_update,params):</fix>
    cur=conn.cursor()
    <fix/>try:
        cur.execute(sql_update,params)
        conn.commit()
    except Exception as err:
        closeDB(conn)
        print(err)
    else: 
        print("update data successfull")
</fix>


#关闭链接
def closeDB(conn):
    conn.close()



