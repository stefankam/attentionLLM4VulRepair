# Sources:
# Building a Chatbot using Telegram and Python (Part 2) by Gareth Dwyer

from time import strftime, time
from pretty_date import prettify_date
import sqlite3

db_name = "robodb.sqlite"
tb_name = "robotb"
mapping = "timestamp INTEGER, name TEXT, location TEXT, description TEXT"
columns = "timestamp, name, location, description"
one_week = 604800    # 1 week
one_year = 31540000  # 1 year
min_data_length = 10 # 10 characters
max_data_length = 70 # 70 characters

# Question mark style does not work here for some unknown reason
# This way of string formatting may be vulnerable to SQL injection

class DBHelper:
    def __init__(self):
        self.db_name = db_name
        self.connection = sqlite3.connect(self.db_name)

    def create_table(self):
        stmt = "CREATE TABLE IF NOT EXISTS {} ({})".format(tb_name, mapping)
        self.connection.execute(stmt)
        self.connection.commit()

    def insert(self, input_row):
        is_valid, violations = self.validate_row(input_row)
        if is_valid:
            name, location, description = input_row
            date = int(time())
            args = (date, name, location, description)
            stmt = "INSERT INTO {} ({}) VALUES {}".format(tb_name, columns, str(args))
            self.connection.execute(stmt)
            self.connection.commit()
        return (is_valid, violations)

    def select_recent(self):
        last = int(time()) - one_week
        stmt = "SELECT {} FROM {} WHERE timestamp >= {} ORDER BY timestamp DESC".format(
            columns, tb_name, str(last))
        rows = self.connection.execute(stmt)
        return rows

    def delete_old(self):
        last = int(time()) - one_year
        stmt = "DELETE FROM {} WHERE timestamp >= {}".format(tb_name, str(last))
        self.connection.execute(stmt)
        self.connection.commit()

    def select_recent_pretty(self):
        return self.prettify_rows(self.select_recent())

    def prettify_rows(self, rows):
        str_builder = ['\n']
        for row in rows:
            str_builder.append("When: {}".format(prettify_date(row[0])))
            str_builder.extend(row[1:])
            str_builder.append('')
        return '\n'.join(str_builder)

    def validate_row(self, input_row):
        <fix/>bad_length = [data for data in input_row if len(data) > max_data_length or len(data) < min_data_length]
        if bad_length:
            return (False, bad_length)</fix>
        return (True, [])
