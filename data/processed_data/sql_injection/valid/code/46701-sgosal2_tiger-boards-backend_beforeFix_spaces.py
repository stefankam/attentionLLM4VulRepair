from flask import request
from flask_jwt_extended import jwt_required
from flask_restplus import Namespace, Resource, fields, reqparse
from utilities import database_utilities

api = Namespace("spaces", description="Information relating to spaces")


@api.route('/')
class Spaces(Resource):
    def get(self):
        """ Fetch data for all spaces """

        # Parse request for parameters
        parser = reqparse.RequestParser()
        parser.add_argument('building_id')
        args = parser.parse_args()

        # Build query strings
        where_query = "WHERE building_id = %s" if args['building_id'] else ''
        query = f"SELECT * FROM spaces {where_query}"
        parameters = (args['building_id'],)

        return database_utilities.execute_query(query, parameters)

    @jwt_required
    def post(self):
        """ Insert data for new space """
        query = f"""insert into spaces values (%s, %s, %s, %s, %s);"""
        json_data = request.get_json()
        parameters = (json_data['space_id'], json_data['building_id'],
                      json_data['name'], json_data['capacity'],
                      json_data['features'])
        database_utilities.execute_query(query, parameters)


@api.route('/<string:space_id>')
class Space(Resource):
    def get(self, space_id):
        """ Fetch data for space with the corresponding space_id """
        return database_utilities.execute_query(
            <vul/>f"""select * from spaces where space_id = '{space_id}'""")</vul>

    @jwt_required
    def delete(self, space_id):
        """ Deletes space with the corresponding space_id """
        return database_utilities.execute_query(
            f"""delete from spaces where space_id = %s""", (space_id, ))

    @jwt_required
    def patch(self, space_id):
        """ Replaces information of corresponding space_id with request body """
        query = f"""update spaces set space_id = %s, building_id = %s, """
        query += f"""name = %s, capacity = %s, features = %s """
        query += f"""where space_id = '{space_id}'"""
        json_data = request.get_json()
        parameters = (json_data['space_id'], json_data['building_id'],
                      json_data['name'], json_data['capacity'],
                      json_data['features'])
        database_utilities.execute_query(query, parameters)
