from app.v1.utils.validator import validate_ints
from app.v2.db.database_config import Database


class BaseModel(Database):
    """ model that defines all models """

    def __init__(self, object_name, table_name):
        self.table_name = table_name
        self.object_name = object_name
        self.error_message = ""
        self.error_code = 200

    def as_json(self):
        pass

    def params_to_values(self, params):
        f = ["'{}'".format(self.escapedString(i)) for i in params]
        return ", ".join(f)

    def escapedString(self, value):
        if isinstance(value, str):
            return value.replace("'", "''")
        return value

    def save(self, fields, *values):
        """ save the object to table """

        query = "INSERT INTO {} ({}) \
        VALUES ({}) RETURNING *".format(
            self.table_name, fields, self.params_to_values(values)
        )
        print(query)
        return super().insert(query)

    def edit(self, key, value, id):
        """ edits a certain column of a table """

        query = "UPDATE {} SET {} = '{}' WHERE id = '{}' \
            RETURNING *".format(
                self.table_name, key, self.escapedString(value), id)

        return self.insert(query)

    def load_all(self):
        """  Get all items in table """

        query = "SELECT * FROM {}".format(self.table_name)

        return self.get_all(query)

    def delete(self, id):
        """ Remove item from table """

        query = "DELETE FROM {} WHERE id = {}".format(self.table_name, id)

        self.execute(query)

    def validate_object(self):
        """This function validates an object and rejects or accepts it"""

        return True

    def find_by(self, key, value):
        """ Find object from table and return """

        query = "SELECT * FROM {} WHERE {} = '{}'".format(
            self.table_name, key, self.escapedString(value))

        data = self.get_one(query)
        if data:
            data[key] = value
        return data
    
    def update_find_by(self, key, value,id):
        """ Find object from table and return """

        <vul/>query = "SELECT * FROM {} WHERE {} = '{}' AND id != {}".format(
            self.table_name, key, self.escapedString(value),id)
</vul>
        data = self.get_one(query)
        if data:
            data[key] = value
        return data

    def find_all_by(self, key, value):
        """ Find objects from table and return """

        query = "SELECT * FROM {} WHERE {} = '{}'".format(
            self.table_name, key, self.escapedString(value))

        data = self.get_all(query)
        return data

    def from_json(self, json):
        return self
