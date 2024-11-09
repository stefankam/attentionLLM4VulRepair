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

        <fix/>sqlName="select count(*) from users where name=%s and password=%s;"
        params = [self.name,self.password]
        checkName=sql.queryDB(self.conn,sqlName,params)</fix>
        result=checkName[0][0]
        if result == 0:
            self.clean()
            return False
        else:
            return True


    def userApply(self):
        <fix/>sql_insert="insert into \</fix>
                users(name,password,email,country,inscription_date) \
                <fix/>values(%s,%s,%s,%s,current_timestamp(0));"</fix>

        <fix/>sqlName="select count(*) from users where name=%s;"
        params = [self.name]
        checkName=sql.queryDB(self.conn,sqlName,params)</fix>
        #no name
        if checkName[0][0] == 0:
            <fix/>params.extend([self.password,self.email,self.country])
            sql.insertDB(self.conn,sql_insert,params)</fix>
            return True
        else:
            return False

    def getUserID(self):
        <fix/>sqlName="select userid from users where name=%s;"
        params = [self.name]
        userid=sql.queryDB(self.conn,sqlName,params)</fix>
        return userid[0][0];

    def getAllPosts(self):
        <fix/>sqlText="select comment from post where userid=%s order by date;"
        params = [self.userid]
        allposts=sql.queryDB(self.conn,sqlName,params)</fix>
        return allposts;


    def getAllComments(self):
        <fix/>sqlText="select comment from comments where userid=%s order by date;"
        params = [self.userid]
        allposts=sql.queryDB(self.conn,sqlText,params)</fix>
        return allposts;

    def getAllInformation(self,userid):
        <fix/>sqlText="select name,password,email,country from users where userid=%s;"
        params = [userid]
        information=sql.queryDB(self.conn,sqlText,params)</fix>
        return information;


    def modifyUserInfo(self,userid,flag):
        sqlText="update users \
                <fix/>set name=%s,password=%s,email=%s,country=%s where userid=%s;"</fix>
        if(flag==1): 
            <fix/>sqlName="select count(*) from users where name=%s;"
            params = [self.name]
            checkName=sql.queryDB(self.conn,sqlName,params)</fix>
            #no name
            if checkName[0][0] == 0:
                <fix/>params.extend([self.password,self.email,self.country,userid])
                sql.updateDB(self.conn,sqlText,params)</fix>
                return True
            else:
                return False
        else:
            <fix/>params=[self.name,self.password,self.email,self.country,userid]
            sql.updateDB(self.conn,sqlText,params)</fix>
            return True;

    def followFriends(self,userid,friendid):
        <fix/>sqlText="insert into friends values(%s,%s);"
        params=[friendid,userid]
        result=sql.insertDB(self.conn,sqlText,params)</fix>
        return result;

    def cancelFollow(self,userid,friendid):
        <fix/>sqlText="delete from friends where userid=%d and friendid=%s;"
        params=[userid,friendid]
        result=sql.deleteDB(self.conn,sqlText,params)</fix>
        return result;

    def getUsers(self,userid):
        sqlText="select userid,name,country,(select Count(*) from friends \
                <fix/>where users.userid=friends.friendid and friends.userid=%s) as follow \
                from users;"
        params=[userid]
        result=sql.queryDB(self.conn,sqlText,params)</fix>
        return result;


    def getUsersByName(self,userid,username):
        sqlText="select userid,name,country,(select Count(*) from friends \
                <fix/>where users.userid=friends.friendid and friends.userid=%s) as follow \
                from users where users.name~%s;"
        params=[userid,username]
        result=sql.queryDB(self.conn,sqlText,params)</fix>
        return result;







