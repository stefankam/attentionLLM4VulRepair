import os
<vul/>from flask import Flask, g, url_for</vul>
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_admin import Admin, helpers
from flask_security import Security
from flask_login import LoginManager
from flask_uploads import patch_request_class, configure_uploads
from werkzeug.utils import find_modules, import_string

from config import app_config

from benwaonline.database import db
from benwaonline.oauth import oauth
from benwaonline.admin import setup_adminviews
from benwaonline.models import user_datastore, User
<vul/>from benwaonline.gallery import gallery
from benwaonline.gallery.forms import images</vul>
from benwaonline.user import user
from benwaonline.auth import auth

FILE_SIZE_LIMIT = 10 * 1024 * 1024

security = Security()
login_manager = LoginManager()

def create_app(config=None):
    app = Flask(__name__)
    app.config.from_object(app_config[config])

    # app.config.update(config or {})
    app.config.from_envvar('BENWAONLINE_SETTINGS', silent=True)
    app.config.from_object('secrets')

    db.init_app(app)
    migrate = Migrate(app, db)
    oauth.init_app(app)

    login_manager.init_app(app)
    @login_manager.user_loader
    def load_user(user_id):
        return User.get(user_id)

    security_ctx = security.init_app(app, user_datastore)
    @security_ctx.context_processor
    def security_context_processor():
        return dict(
            admin_base_template=admin.base_template,
            admin_view=admin.index_view,
            h=helpers,
            get_url=url_for
        )

    admin = Admin(app, name='benwaonline', template_mode='bootstrap3')
    setup_adminviews(admin, db)

    register_blueprints(app)
    register_cli(app)
    register_teardowns(app)

    app.register_blueprint(gallery)
    app.register_blueprint(auth)
    app.register_blueprint(user)

    configure_uploads(app, (images,))
    patch_request_class(app, FILE_SIZE_LIMIT)

    return app

def register_blueprints(app):
    """
    Register all blueprint modules
    Reference: Armin Ronacher, "Flask for Fun and for Profit" PyBay 2016.
    """
    for name in find_modules('benwaonline.blueprints'):
        mod = import_string(name)
        if hasattr(mod, 'bp'):
            app.register_blueprint(mod.bp)

    return None

def register_cli(app):
    @app.cli.command('initdb')
    def initdb_command():
        """Creates the database tables."""
        init_db()
        print('Initialized the database.')

def init_db():
    import benwaonline.models
    db.create_all()

def register_teardowns(app):
    @app.teardown_appcontext
    def close_db(error):
        """Closes the database again at the end of the request."""
        if hasattr(g, 'sqlite_db'):
            g.sqlite_db.close()

