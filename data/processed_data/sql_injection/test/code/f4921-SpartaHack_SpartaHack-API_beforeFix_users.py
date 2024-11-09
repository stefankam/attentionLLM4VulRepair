from flask_restful import Resource
from werkzeug.exceptions import BadRequest
from flask import request,jsonify,g
from datetime import datetime
from sqlalchemy import exists,and_
from sqlalchemy.orm.exc import NoResultFound
from common.json_schema import User_Schema
from common.utils import headers,is_logged_in,has_admin_privileges
from common.utils import bad_request,unauthorized,forbidden,not_found,internal_server_error,unprocessable_entity,conflict

class Users_RUD(Resource):
    def get(self,user_id):
        <vul/>pass</vul>

    def put(self,user_id):
        pass

    def delete(self,user_id):
        pass

class Users_CR(Resource):
    <vul/>def get(self):</vul>
        pass

    def post(self):
        pass