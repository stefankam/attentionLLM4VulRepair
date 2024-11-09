import os

BASE = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    BASE_DIR = BASE
    SQLALCHEMY_MIGRATE_REPO = os.path.join(BASE_DIR, 'db_repository')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOADED_IMAGES_DEST = '/static/'
    <fix/>UPLOADED_BENWA_DIR = os.path.join(BASE, 'benwaonline', 'static', 'tempbenwas')
    REDIRECT_BACK_DEFAULT = 'gallery.show_posts'</fix>

class DevConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE, 'db', 'benwaonline.db')
    DEBUG = True
    SECRET_KEY = 'not-so-secret'

class TestConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'sqlite://'
    TESTING = True
    TWITTER_CONSUMER_KEY = 'consume'
    TWITTER_CONSUMER_SECRET = 'secret'
    WTF_CSRF_ENABLED = False
    SECRET_KEY = 'not-so-secret'

class ProdConfig(Config):
    DEBUG = False

app_config = {
    'dev': DevConfig,
    'test': TestConfig,
    'prod': ProdConfig
}
