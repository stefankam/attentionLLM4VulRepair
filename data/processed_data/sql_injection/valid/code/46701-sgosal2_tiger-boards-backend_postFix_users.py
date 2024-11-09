from flask import request
from flask_jwt_extended import jwt_required
from flask_restplus import Namespace, Resource, fields
from utilities import database_utilities

api = Namespace("users", description="Information relating to users")


@api.route('/')
class Users(Resource):
    def get(self):
        """ Fetch data for all users """
        return database_utilities.execute_query("select * from users")

    @jwt_required
    def post(self):
        """ Insert data for new users """
        query = f"""insert into users values (%s);"""
        json_data = request.get_json()
        parameters = (json_data['user_id'], )
        database_utilities.execute_query(query, parameters)


@api.route('/<string:user_id>')
class User(Resource):
    def get(self, user_id):
        """ Fetch data for user with corresponding user_id """
        <fix/>return database_utilities.execute_query(f"""select * from users where user_id = %s""", (user_id, ))</fix>

    @jwt_required
    def delete(self, user_id):
        """ Deletes user with the corresponding user_id """
        <fix/>return database_utilities.execute_query(f"""delete from users where user_id = %s""", (user_id, ))</fix>

    @jwt_required
    def patch(self, user_id):
        """ Replaces information of corresponding user_id with request body """
        query = f"""update users set user_id = %s """
        <fix/>query += f"""where user_id = %s"""</fix>
        json_data = request.get_json()
        <fix/>parameters = (json_data['user_id'], user_id)</fix>
        database_utilities.execute_query(query, parameters)
