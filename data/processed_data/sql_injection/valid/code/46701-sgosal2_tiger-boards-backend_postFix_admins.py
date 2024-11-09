from flask import request
from flask_jwt_extended import jwt_required
from flask_restplus import Namespace, Resource, fields
from utilities import database_utilities

api = Namespace("admins", description="Information relating to system admins")


@api.route('/')
class Admins(Resource):
    def get(self):
        """ Fetch data for all admins """
        return database_utilities.execute_query("select * from admins")

    @jwt_required
    def post(self):
        """ Insert data for a new admin """
        query = f"""insert into admins values (%s);"""
        json_data = request.get_json()
        parameters = (json_data['email'], )
        database_utilities.execute_query(query, parameters)


@api.route('/<string:email>')
class Admin(Resource):
    def get(self, email):
        """ Fetch data for admin with the corresponding email """
        <fix/>return database_utilities.execute_query(f"""select * from admins where email = %s""", (email, ))</fix>

    @jwt_required
    def delete(self, email):
        """ Deletes admin with the corresponding email """
        <fix/>return database_utilities.execute_query(f"""delete from admins where email = %s""", (email, )</fix>)
