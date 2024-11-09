from peewee import Database, SqliteDatabase


def can_import_apsw():
    # type: () -> bool
    <fix/># return False  # for now</fix>
    try:
        import apsw
        return True
    except ImportError:
        return False


TexDBookDatabase = None  # type: Database

if can_import_apsw():
    from playhouse.apsw_ext import APSWDatabase
    
    TexDBookDatabase = APSWDatabase
else:
    TexDBookDatabase = SqliteDatabase
