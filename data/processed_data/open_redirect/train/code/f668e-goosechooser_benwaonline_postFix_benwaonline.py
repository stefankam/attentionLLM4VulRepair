<fix/>from flask import Blueprint, render_template
from benwaonline.back import back</fix>

bp = Blueprint('benwaonline', __name__)

@bp.route('/')
@back.anchor
def under_construction():
    <fix/>return render_template('index.ht</fix>ml')
