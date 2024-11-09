from flask import request, jsonify
from flask_jwt_extended import (
    create_access_token, create_refresh_token, set_access_cookies,
    set_refresh_cookies, jwt_refresh_token_required, get_jwt_identity
)
from flask_restplus import Namespace, Resource
from utilities import database_utilities

api = Namespace("login", description="Endpoint used to obtain JWT")


@api.route('/')
class Login(Resource):
    def post(self):
        """ Returns JWT upon login verification """
        json_data = request.get_json()
        if not json_data['email']:
            return jsonify({"msg": "Missing email"}), 400

        data = database_utilities.execute_query(
            <vul/>f"""select * from admins where email = '{json_data['email']}'""")</vul>
        if data:
            email = data[0]['email']
            access_token = create_access_token(identity=email)
            refresh_token = create_refresh_token(identity=email)

            resp = jsonify({"login": True})
            set_access_cookies(resp, access_token)
            set_refresh_cookies(resp, refresh_token)
            return resp
        else:
            return jsonify({"msg": "User is not an admin"})


@api.route('/refresh')
class Refresh(Resource):
    @jwt_refresh_token_required
    def post(self):
        current_user = get_jwt_identity()
        access_token = create_access_token(identity=current_user)

        resp = jsonify({"refresh": True})
        set_access_cookies(resp, access_token)
        return resp
