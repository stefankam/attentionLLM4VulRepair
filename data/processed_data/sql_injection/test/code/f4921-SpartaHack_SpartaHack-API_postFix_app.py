from flask import Flask, jsonify, make_response,request, g
from flask_restful import Api
from celery import Celery
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from common.utils import unauthorized,headers,not_found

from config import load_env_variables, DevelopmentConfig, ProdConfig

#loading environment variables
load_env_variables()

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)#loading config data into flask app from config object.
api = Api(app)

#reflecting classes
print("Reflecting classes...")
Base = automap_base()
engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"],pool_size=20,max_overflow=20,pool_pre_ping=True)
Base.prepare(engine, reflect=True)
print("Classes reflected...")


@app.before_request
def create_session():
    """
    Before processing any request. Create a session by checking out a connection from the connection pool.
    <fix/>Also set global variables to be accessed for the life time of the request</fix>
    """
    g.session = Session(engine)
    g.Base = Base

@app.after_request
def commit_and_close_session(resp):
    """
    After all the processing is done. Commit the changes and close the session to return the connection object back to the connection pool.
    """
    g.session.commit()
    g.session.close()
    return resp

#loading resources
from resources.faqs import Faqs_RUD, Faqs_CR
from resources.announcements import Announcements_RUD, Announcements_CR
from resources.hardware import Hardware_RUD, Hardware_CR
from resources.sponsors import Sponsor_RD, Sponsor_CR
from resources.schedule import Schedule_RUD, Schedule_CR
from resources.applications import Applications_RU, Applications_CR
from resources.users import Users_RUD, Users_CR

@api.representation('application/json')
def ret_json(data, code, headers=None):
    """
    Create proper request object based on the return dictionary.
    """
    if code == 204:
        resp = make_response('', code)
    else:
        resp = make_response(jsonify(data), code)
    resp.headers.extend(headers)
    return resp

#might only need this for email sending so that email sending does not clog up the resources
task_queue=Celery("SpartaHack_API_2019",broker=app.config["CELERY_BROKER_URL"])

#adding resources. Just flask-restful things :)
api.add_resource(Faqs_RUD,"/faqs/<int:faq_id>")
api.add_resource(Faqs_CR,"/faqs")
api.add_resource(Announcements_RUD,"/announcements/<int:announcement_id>")
api.add_resource(Announcements_CR,"/announcements")
api.add_resource(Hardware_RUD,"/hardware/<int:hardware_id>")
api.add_resource(Hardware_CR,"/hardware")
api.add_resource(Sponsor_RD,"/sponsors/<int:sponsor_id>")
api.add_resource(Sponsor_CR,"/sponsors")
api.add_resource(Schedule_RUD,"/schedule/<int:schedule_id>")
api.add_resource(Schedule_CR,"/schedule")
api.add_resource(Applications_RU,"/applications/<int:application_id>")
api.add_resource(Applications_CR,"/applications")
api.add_resource(Users_RUD,"/users/<int:user_id>")
api.add_resource(Users_CR,"/users")


@app.route("/")
def helloworld():
    """
    For flask app test and general info about the API.
    Will also be used to check if the api is live or not on the slack hook
    """
    metadata = {
                "Organization":"SpartaHack",
                "Backend Developers":"Yash, Jarek",
                "Frontend Developers":"Harrison, Jessica, Jarek",
                "Contact":"hello@spartahack.com",
                "Version":"0.5.0"
               }
    return (metadata,200,headers)


if __name__ == '__main__': #running on local server. This needs to change for prod
    app.run(debug=True)
