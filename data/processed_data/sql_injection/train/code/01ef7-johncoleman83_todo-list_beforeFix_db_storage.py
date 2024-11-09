#!/usr/bin/python3
"""
Database storage engine
"""
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, scoped_session
from models.base_model import Base
<vul/>from models import base_model, task, user</vul>
from models.secrets import USER, PW, HOST, DB

class DBStorage:
    """
    handles long term storage of all class instances
    """
    CNC = {
        'Task': task.Task,
        <vul/>'User': user.User,</vul>
    }
    __engine = None
    __session = None

    def __init__(self):
        """
        creates the engine self.__engine
        """
        self.__engine = create_engine(
            'mysql+mysqldb://{}:{}@{}:3306/{}'
            .format(USER, PW, HOST, DB)
        )

    def all(self, cls=None):
        """
        returns a dictionary of all objects
        """
        obj_dict = {}
        if cls is not None:
            a_query = self.__session.query(DBStorage.CNC[cls])
            for obj in a_query:
                obj_ref = "{}.{}".format(type(obj).__name__, obj.id)
                if cls == 'User' and obj.tasks:
                    pass
                obj_dict[obj_ref] = obj
            return obj_dict

        for c in DBStorage.CNC.values():
            a_query = self.__session.query(c)
            for obj in a_query:
                obj_ref = "{}.{}".format(type(obj).__name__, obj.id)
                if type(c).__name__ == 'User' and obj.tasks:
                    pass
                obj_dict[obj_ref] = obj
        return obj_dict

    def new(self, obj):
        """
            adds objects to current database session
        """
        self.__session.add(obj)

    def save(self):
        """
            commits all changes of current database session
        """
        self.__session.commit()

    def rollback_session(self):
        """
            rollsback a session in the event of an exception
        """
        self.__session.rollback()

    def delete(self, obj=None):
        """
            deletes obj from current database session if not None
        """
        if obj:
            self.__session.delete(obj)
            self.save()

    def delete_all(self):
        """
           deletes all stored objects, for testing purposes
        """
        for c in DBStorage.CNC.values():
            a_query = self.__session.query(c)
            all_objs = [obj for obj in a_query]
            for obj in range(len(all_objs)):
                to_delete = all_objs.pop(0)
                to_delete.delete()
        self.save()

    def reload(self):
        """
           creates all tables in database & session from engine
        """
        Base.metadata.create_all(self.__engine)
        self.__session = scoped_session(
            sessionmaker(
                bind=self.__engine,
                expire_on_commit=False))

    def close(self):
        """
            calls remove() on private session attribute (self.session)
        """
        self.__session.remove()

    def get(self, cls, id):
        """
        retrieves one object based on class name and id
        """
        if cls and id:
            fetch = "{}.{}".format(cls, id)
            all_obj = self.all(cls)
            return all_obj.get(fetch)
        return None

    def count(self, cls=None):
        """
            returns the count of all objects in storage
        """
        return (len(self.all(cls)))
