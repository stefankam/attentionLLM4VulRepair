import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

class DatabaseController(object):
        def __init__(self, connection_address, connection_port, user_name, password, database):
                self.connection = mysql.connector.connect(host = connection_address, user=user_name, password = password, db = database)
                self.cursor = self.connection.cursor(buffered=True)
        <fix/>def create_table(self):</fix>
                query = "CREATE TABLE IF NOT EXISTS DPNET(why_mySQL int)"
                self.cursor.execute(query)
                self.connection.commit()
        <fix/>def destroy_table(self):</fix>
                query = "DROP TABLE IF EXISTS DPNET"
                self.cursor.execute(query)
                self.connection.commit()

        def verify_account(self, email, user_password):
                <fix/>#query = "SELECT Pass FROM user WHERE Email = '" + email +"'"
                query = "SELECT Pass FROM user WHERE Email = %s"
                self.cursor.execute(query, (email, ))</fix>
                fetch = self.cursor.fetchone()
                password = " ".join(map(str, fetch))
                return check_password_hash(password, user_password)

        def set_report(self, reportID, userID, summary, description):
                <fix/>#query = "INSERT INTO `testdb`.`report` (`Report_ID`, `User_ID`, `Summary`, `Description`, `Votes`, `Is_Resolved`) VALUES ('" + reportID + "', '" + userID + "', '" + summary + "', '" + description + "', '0', '0')"
                query = "INSERT INTO `testdb`.`report` (`Report_ID`, `User_ID`, `Summary`, `Description`, `Votes`, `Is_Resolved`) VALUES (%S, %s, %s, %s, '0', '0')"
                self.cursor.execute(query, (reportID, userID, summary, description))</fix>
                self.connection.commit()

        def increment_vote(self, reportID):
                <fix/><fix/>#query1 = "SELECT Votes FROM report WHERE Report_ID = '" + reportID +"'"
                query1 = "SELECT Votes FROM report WHERE Report_ID = %s"
                self.cursor.execute(query1, (reportID, ))</fix></fix>
                fetch = self.cursor.fetchone()
                curVote = " ".join(map(str, fetch))
                intVote = int(curVote)
                intVote = intVote + 1
                <fix/>#query2 = "UPDATE `testdb`.`report` SET `Votes` = '" + str(intVote) + "' WHERE `report`.`Report_ID` = " + reportID
                query2 = "UPDATE `testdb`.`report` SET `Votes` = '" + str(intVote) + "' WHERE `report`.`Report_ID` = %s"
                self.cursor.execute(query2, (reportID, ))</fix>
                self.connection.commit()

        def get_vote(self, reportID):
                #query1 = "SELECT Votes FROM report WHERE Report_ID = '" + reportID +"'"
                query1 = "SELECT Votes FROM report WHERE Report_ID = %s"
                self.cursor.execute(query1, (reportID, ))
                fetch = self.cursor.fetchone()
                curVote = " ".join(map(str, fetch))
                intVote = int(curVote)
                return intVote

        def resolve_issue(self, reportID):
                <fix/>#query = "UPDATE `testdb`.`report` SET `Is_Resolved` = '1' WHERE `report`.`Report_ID` = " + reportID
                query = "UPDATE `testdb`.`report` SET `Is_Resolved` = '1' WHERE `report`.`Report_ID` = %s"
                self.cursor.execute(query, (reportID, ))</fix>
                self.connection.commit()

        def get_report(self, reportID):
                <fix/>#query = "SELECT * FROM report WHERE Report_ID = " + reportID 
                query = "SELECT * FROM report WHERE Report_ID = %s"
                self.cursor.execute(query, (reportID, ))</fix>
                self.connection.commit()
                fetch = self.cursor.fetchone()
                report = " ".join(map(str, fetch))
                return report

        def create_basic_user(self, userID, fName, lName, email, password):
                password2 = generate_password_hash(password)
                <fix/>#query = "INSERT INTO `testdb`.`user` (`ID`, `FName`, `LName`, `Email`, `Pass`, `Role`) VALUES ('" + userID + "', '" + fName + "', '" + lName + "', '" + email +"', '" + password2 + "', '0')"
                query = "INSERT INTO `testdb`.`user` (`ID`, `FName`, `LName`, `Email`, `Pass`, `Role`) VALUES (%s, %s, %s, %s, %s, '0')"
                self.cursor.execute(query, (userID, fName, lName, email, password2))</fix>
                self.connection.commit()

        def get_basic_user(self, userMail):
                #query = "SELECT * FROM user WHERE EMail = " + userMail
                query = "SELECT * FROM user WHERE EMail = %s"
                self.cursor.execute(query, (userMail, ))
                self.connection.commit()
                fetch = self.cursor.fetchone()
                userData = " ".join(map(str, fetch))
                return userData

        def create_faculty_user(self, userID, fName, lName, email, password):
                password2 = generate_password_hash(password)
                <fix/>#query = "INSERT INTO `testdb`.`user` (`ID`, `FName`, `LName`, `Email`, `Pass`, `Role`) VALUES ('" + userID + "', '" + fName + "', '" + lName + "', '" + email +"', '" + password2 + "', '1')"
                query = "INSERT INTO `testdb`.`user` (`ID`, `FName`, `LName`, `Email`, `Pass`, `Role`) VALUES (%s, %s, %s, %s, %S, '1')"
                self.cursor.execute(query, (userID, fName, lName, email, password2))</fix>
                self.connection.commit()
        
        def close_connection(self):
                self.connection.close()
        
<fix/>#dbc = DatabaseController('localhost', 3306, 'testuser', 'test623', 'testdb')
#dbc.create_basic_user("1586390", "Daniel", "Gonzalez", "dgonz023@fiu.edu", "dpnet")
#print(dbc.verify_account("dgonz023@fiu.edu", "dpnet"))</fix>

