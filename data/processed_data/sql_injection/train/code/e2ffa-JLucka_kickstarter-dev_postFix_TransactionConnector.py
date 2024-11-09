import datetime

import MySQLdb

from backend.SQLConnector import SQLConnector

TABLE_NAME = "transactions"


class Transaction:
    def __init__(self, project_id, user_id, money):
        self.project_id = project_id
        self.user_id = user_id
        self.money = money
        self.time = datetime.datetime.now().isoformat(' ')

    def to_json_obj(self):
        obj = {
            'id': self.id,
            'project_id': self.project_id,
            'user_id': self.user_id,
            'money': self.money,
            'time': self.time
            }
        return obj

    def to_database_query(self):
        data = [self.project_id, self.user_id, self.money, self.time]
        data = [repr(x) for x in data]
        labels = ["project_id", "user_id", "money", "timestamp"]
        return dict(zip(labels, data))


class TransactionConnector(SQLConnector):
    def __init__(self):
        SQLConnector.__init__(self)
        self.table_name = TABLE_NAME

    def insert_into(self, transaction):
        return SQLConnector.insert_into(self, transaction.to_database_query())

    def support_project(self, user_id, project_id, money):
        try:
            if self.can_user_pass_that_amount_of_money(user_id, money) \
                    and self.check_if_this_project_is_in_database(project_id):
                self.save_accepted_transaction(user_id, project_id, money)
                return True
            else:
                self.save_failure_transaction(user_id, project_id, money)
        except MySQLdb.Error:
            self.db.rollback()
        return False

    def save_accepted_transaction(self, user_id, project_id, money):
        <fix/>self.cursor.execute("update users set money = money - %s where id = %s", (money, user_id))
        self.cursor.execute("update projects set money = money + %s where id = %s", (money, project_id))
        self.cursor.execute("insert into transactions (project_id, user_id, money, timestamp, state) values (%s, %s, "
                            "%s, now(), 'accepted' )", (project_id, user_id, money))</fix>
        self.db.commit()

    def save_failure_transaction(self, user_id, project_id, money):
        <fix/>self.cursor.execute("insert into transactions (project_id,user_id, money, timestamp, state) values (%s, %s, "
                            "%s, now(), 'failed' )", (project_id, user_id, money))</fix>
        self.db.commit()

    def check_if_this_project_is_in_database(self, project_id):
        <fix/>self.cursor.execute("SELECT count(id) FROM projects where id = %s", (project_id,))</fix>
        return self.cursor.fetchall()[0][0] == 1

    def can_user_pass_that_amount_of_money(self, user_id, money):
        <fix/>self.cursor.execute("SELECT count(id) FROM kickstarter.users where id = %s and money >= %s", (user_id, money))</fix>
        return self.cursor.fetchall()[0][0]
