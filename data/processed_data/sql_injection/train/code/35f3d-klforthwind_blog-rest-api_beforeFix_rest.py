from flask import Flask
from flask_restful import Api, Resource, reqparse
from WebHandler import getHTML

app = Flask(__name__)
api = Api(app)

# Blog REST API
class Blog(Resource):

    #GET Request- Returns website in full html
    def get(self, name):
        <vul/>return getHTML(name)</vul>
        
    #def post(self, name):
    #def put(self, name):
    #def delete(self, name):

# Access the api from 198.58.107.98:6969/blog/url-name
api.add_resource(Blog, "/blog/<string:name>")

app.run(host='198.58.107.98', port=6969, debug=True)