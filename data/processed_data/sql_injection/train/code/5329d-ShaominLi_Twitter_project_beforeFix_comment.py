from modules import sql


class Comment:
    def __init__(self,conn):
        self.conn=conn;
    
    def getCommentsByUser(self,userid):
        <vul/>sqlText="select comment from comments order by date desc where userid=%d"%(userid)
        result=sql.queryDB(self.conn,sqlText)</vul>
        return result;
    
    def getCommentsByPostid(self,postid,userid):
        <vul/>sqlText="select (select Count(*) from comment_like where comments.commentid = comment_like.commentid) as like,(select Count(*) from comment_like where comments.commentid = comment_like.commentid and comment_like.userid=%d) as flag,commentid,name,comment from users,comments where users.userid=comments.userid and postid=%d order by date desc;"%(userid,postid)
        result=sql.queryDB(self.conn,sqlText)</vul>
        return result;

    def getCommentsLike(self,commentid):
        <vul/>sqlText="select userid from comment_like where commentid=%d"%(commentid)
        result=sql.queryDB(self.conn,sqlText)</vul>
        return result;
	
    def insertData(self,comment,userid,postid):
        <vul/>sqlText="insert into comments(comment,userid,date,postid) values('%s',%d,current_timestamp(0),%d);"%(comment,userid,postid)
        result=sql.insertDB(self.conn,sqlText)</vul>
        return result;

    def deleteComment(self,commentid):
        <vul/>sqlText="delete from comments where commentid=%d"%(commentid)
        result=sql.deleteDB(self.conn,sqlText)</vul>
        return result;

    def likeComments(self,commentid,userid):
        <vul/>sqlText="insert into comment_like values(%d,%d);"%(userid,commentid)
        result=sql.insertDB(self.conn,sqlText)</vul>
        return result;

    def dislikeComments(self,commentid,userid):
        <vul/>sqlText="delete from comment_like where commentid=%d and userid=%d;"%(commentid,userid)
        result=sql.deleteDB(self.conn,sqlText)</vul>
        return result;



