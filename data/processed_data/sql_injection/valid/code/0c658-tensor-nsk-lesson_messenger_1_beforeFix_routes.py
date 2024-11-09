from flask import Blueprint, request, jsonify
from modules.ProfileManager.api.db_methods import db_delProfile, db_getProfileInfo, db_getProfilesInfo, db_updateProfileInfo
from modules.ProfileManager.api.db_methods import db_FullDelProfile, db_isProfileExists
from modules.ProfileManager.api.functions import isProfileDeleted, isProfileBlocked
import json

profile_module = Blueprint('profile', __name__)

<vul/>@profile_module.route('/profile/<int:ID>', methods=['GET', 'PUT', 'DELETE'])</vul>
def profile(ID):
    if not db_isProfileExists(ID):
        return jsonify({'status': 1, 'message': 'Такого аккаунта не существует'})

    if request.method == 'GET':
        return jsonify(db_getProfileInfo(ID))

    else:
        if request.method == 'PUT':
            <vul/>data = json.loads(request.data)</vul>
            if isProfileDeleted(ID):
                return jsonify({'status': 0, 'message': 'Невозможно изменить данные удалённого аккаунта'})

            if isProfileBlocked(ID):
                return jsonify({'status': 0, 'message': 'Невозможно изменить данные заблокированного'})

            return jsonify(db_updateProfileInfo(ID, data))

        elif request.method == 'DELETE':
            if isProfileDeleted(ID):
                return jsonify({'status': 0, 'message': 'Аккаунт уже удалён'})

            #return db_delProfile(ID, status=True)
            return jsonify(db_FullDelProfile(ID))


<vul/>@profile_module.route('/profiles')</vul>
def profiles():
    return jsonify(db_getProfilesInfo())