from modules import sql


class Comment:
    def __init__(self,conn):
        self.conn=conn;
    
    def getCommentsByUser(self,userid):
        <fix/>sqlText="select comment from comments order by date desc where userid=%s"
        params=[userid]
        result=sql.queryDB(self.conn,sqlText,params)</fix>
        return result;
    
    def getCommentsByPostid(self,postid,userid):
        <fix/>sqlText="select (select Count(*) from comment_like where \
        comments.commentid = comment_like.commentid) as like,(select Count(*) \
                from comment_like where comments.commentid = \
                comment_like.commentid and comment_like.userid=%s) as \
                flag,commentid,name,comment from users,comments where \
                users.userid=comments.userid and postid=%s order by date desc;"
        params=[userid,postid]
        result=sql.queryDB(self.conn,sqlText,params)</fix>
        return result;

    def getCommentsLike(self,commentid):
        <fix/>sqlText="select userid from comment_like where commentid=%s"
        params=[commentid]
        result=sql.queryDB(self.conn,sqlText,params)</fix>
        return result;
	
    def insertData(self,comment,userid,postid):
        <fix/>sqlText="insert into comments(comment,userid,date,postid) \
        values(%s,%s,current_timestamp(0),%s);"
        params=[comment,userid,postid]
        result=sql.insertDB(self.conn,sqlText,params)</fix>
        return result;

    def deleteComment(self,commentid):
        <fix/>sqlText="delete from comments where commentid=%s"
        params=[commentid]
        result=sql.deleteDB(self.conn,sqlText,params)</fix>
        return result;

    def likeComments(self,commentid,userid):
        <fix/>sqlText="insert into comment_like values(%s,%s);"
        params=[userid,commentid]
        result=sql.insertDB(self.conn,sqlText,params)</fix>
        return result;

    def dislikeComments(self,commentid,userid):
        <fix/>sqlText="delete from comment_like where commentid=%s and userid=%s;"
        params=[commentid,userid]
        result=sql.deleteDB(self.conn,sqlText,params)</fix>
        return result;



