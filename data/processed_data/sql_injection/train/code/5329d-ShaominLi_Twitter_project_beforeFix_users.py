from modules import sql

class Users:
    def __init__(self,conn=None,name=None,password=None,email=None,country=None):
        self.name=name
        self.password=password
        self.email=email
        self.country=country
        self.conn=conn

    def clean(self):
        self.name=None;
        self.password=None;
        self.email=None;
        self.count=None;
 

    def userLogin(self):

        <vul/>sqlName="select count(*) from users where name='%s' and \
                password='%s';"%(self.name,self.password)
        checkName=sql.queryDB(self.conn,sqlName)
</vul>
        result=checkName[0][0]
        if result == 0:
            self.clean()
            return False
        else:
            return True


    def userApply(self):
        <vul/>t_sql_insert="insert into \</vul>
                users(name,password,email,country,inscription_date) \
                <vul/>values('{name}','{psw}','{email}','{country}',current_timestamp(0));"
        sql_insert=t_sql_insert.format(name=self.name,psw=self.password,\
                email=self.email,country=self.country)</vul>

        <vul/>sqlName="select count(*) from users where name='%s';"%(self.name)
        checkName=sql.queryDB(self.conn,sqlName)
</vul>    
        #no name
        if checkName[0][0] == 0:
            <vul/>sql.insertDB(self.conn,sql_insert)</vul>
            return True
        else:
            return False

    def getUserID(self):
        <vul/>sqlName="select userid from users where name='%s';"%(self.name)
        userid=sql.queryDB(self.conn,sqlName)</vul>
        return userid[0][0];

    def getAllPosts(self):
        <vul/>sqlText="select comment from post where userid=%d order by date;"
        allposts=sql.queryDB(self.conn,sqlText)</vul>
        return allposts;


    def getAllComments(self):
        <vul/>sqlText="select comment from comments where userid=%d order by date;"
        allposts=sql.queryDB(self.conn,sqlText)</vul>
        return allposts;

    def getAllInformation(self,userid):
        <vul/>sqlText="select name,password,email,country from users where userid=%d;"%(userid)
        information=sql.queryDB(self.conn,sqlText)</vul>
        return information;


    def modifyUserInfo(self,userid,flag):
        sqlText="update users \
                <vul/>set name='%s',password='%s',email='%s',country='%s' \
                where userid='%d';"%(self.name,self.password,self.email,self.country,userid)</vul>
        if(flag==1): 
            <vul/>sqlName="select count(*) from users where name='%s';"%(self.name)
            checkName=sql.queryDB(self.conn,sqlName)</vul>
            #no name
            if checkName[0][0] == 0:
                <vul/><vul/>sql.updateDB(self.conn,sqlText)</vul></vul>
                return True
            else:
                return False
        else:
            sql.updateDB(self.conn,sqlText)
            return True;

    def followFriends(self,userid,friendid):
        <vul/>sqlText="insert into friends values(%d,%d);"%(friendid,userid)
        result=sql.insertDB(self.conn,sqlText)</vul>
        return result;

    def cancelFollow(self,userid,friendid):
        <vul/>sqlText="delete from friends where userid=%d and friendid=%d;"%(userid,friendid)
        result=sql.deleteDB(self.conn,sqlText)</vul>
        return result;

    def getUsers(self,userid):
        sqlText="select userid,name,country,(select Count(*) from friends \
                <vul/>where users.userid=friends.friendid and friends.userid=%d) as follow \
                from users;"%(userid)
        result=sql.queryDB(self.conn,sqlText)</vul>
        return result;


    def getUsersByName(self,userid,username):
        sqlText="select userid,name,country,(select Count(*) from friends \
                <vul/>where users.userid=friends.friendid and friends.userid=%d) as follow \
                from users where users.name='%s';"%(userid,username)
        result=sql.queryDB(self.conn,sqlText)</vul>
        return result;







