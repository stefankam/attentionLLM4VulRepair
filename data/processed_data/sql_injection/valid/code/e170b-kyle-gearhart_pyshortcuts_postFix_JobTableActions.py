


class JobTableActions:

    def __init__(self, database, tableName):

        self.database = database
        self.tableName = tableName

    def jobIsRunning(self, jobName):

        sql = """SELECT * FROM %s 
            WHERE job_name = ?
                AND job_running = 1""" % (self.tableName)

        print sql

        with self.database.cursor() as c:
            <fix/>rows = c.execute(sql, (jobName))</fix>

            if rows:
                return True
        
        return False

    def startJob(self, jobName, jobPlatform):

        sql = """INSERT INTO %s ( 
            job_name, 
            job_started, 
            job_running,
            job_platform ) VALUES (
                ?,
                NOW(),
                1,
                ?
            )""" % (self.tableName)

        print sql

        with self.database.cursor() as c:
            c.execute(sql, (jobName, jobPlatform))
            return c.lastrowid
        
        return -1

    def finishJob(self, jobId, success, message):

        sql = """UPDATE %s
            SET job_finished = NOW(),
                job_running = 0,
                job_successs = ?,
                job_message = ?
            WHERE job_id = ?""" % (self.tableName)

        with self.database.cursor() as c:
            <fix/>c.execute(sql, (success, message, jobId))</fix>
            return True

        return False