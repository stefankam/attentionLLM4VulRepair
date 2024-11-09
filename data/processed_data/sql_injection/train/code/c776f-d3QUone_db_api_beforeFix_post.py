__author__ = 'vladimir'

import ujson

from flask import Blueprint

<vul/>BASE_URL = "/post"</vul>

<vul/>post = Blueprint("post", __name__)</vul>


<vul/>@post.route(BASE_URL + "/create", methods=["GET"])</vul>
def create():
    return ujson.dumps({"success": True})
