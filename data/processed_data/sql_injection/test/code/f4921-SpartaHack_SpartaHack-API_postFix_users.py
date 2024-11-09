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
    """
    For GET UPDATE and DELETE for specific user_id
    """
    def get(self,user_id):
        <fix/>"""
        GET the user details based on specific user_id
        We are considering user and application separate entities. Previously, a user submitted the application details at the time of user account creation.
        Now, the user creates an account and then logs in to submit the application.
        Compared to old API we are not returning the application and rsvp details with the user details, rather we are returning the application_id and rsvp_id to help the front-end make GET request on application and rsvp endpoint in a separate requests.
        """
        #using get instead of query and it is marginally faster than filter
        #check for multiple entries need to be done at POST and not during GET or PUT or DELETE
        user_status,calling_user = has_admin_privileges()
        if user_status == "no_auth_token":
            return (bad_request,400,headers)

        if user_status == "not_logged_in":
            return (unauthorized,401,headers)

        try:
            user = g.session.query(g.Base.classes.users).get(user_id)
        except Exception as err:
            print(type(err))
            print(err)
            return (internal_server_error,500,headers)
        # *Only allow directors, organizers and users calling
        if user_status in ["director","organizer"] or calling_user == user:
            if user:
                ret = User_Schema().dump(user).data
                # *<class_name>_collection is way by which SQLAlchemy stores relationships.
                # *The collection object contains one related object in one-to-one relationship and more than one object in one-to-many relationships
                # *set application_id if application object is found in the applications_collection i.e. if the user has submitted the application
                # *This is different than old API in the sense that a user does not necessarily have a application at the time of account creation. User has the option of submitting the application later on.
                ret["application_id"] = user.applications_collection[0].id if len(user.applications_collection)>0 else None
                # *set rsvp_id if rsvp object is found in the rsvps_collection i.e. if the user has rsvped
                ret["rsvp_id"] = user.rsvps_collection[0].id if len(user.rsvps_collection)>0 else None
                return (ret,200,headers)
            else:
                return (not_found,404,headers)
        else:
            return(forbidden,403,headers)</fix>

    def put(self,user_id):
        pass

    def delete(self,user_id):
        pass

class Users_CR(Resource):
    <fix/>def post(self):</fix>
        pass

    <fix/>def get(self):
        """
        GET all the users at a time.
        Application_id and rsvp_id is not gonna be returned when GET is called on all the users.
        Because getting the application_id and rsvp_id makes db calls and making calls for application_id and rsvp_id on hundreds of users is costly.
        """
        try:
            all_users = g.session.query(g.Base.classes.users).all()
            ret = User_Schema(many = True).dump(all_users).data
            return (ret,200,headers)
        except Exception as err:
            print(type(err))
            print(err)
            return (internal_server_error,500,headers)</fix>
