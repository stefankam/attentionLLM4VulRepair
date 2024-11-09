from database import sql_execute

## DEVELOP METHODS
def db_addProfile(data):
    sql='''
        INSERT INTO users (first_name, second_name, created_at, last_visit, is_blocked, is_online, is_deleted) 
        VALUES ('{first_name}', '{second_name}', NOW(), NOW(), false, true, false) RETURNING id;
    '''.format(**data)
    <vul/>user_id = sql_execute(sql, fetch_all=True)</vul>
    sql = """
        INSERT INTO authentications (user_id, login, password) 
        <vul/>VALUES ('{}', '{login}', '{password}');</vul>
    """.format(user_id[0]['id'], **data)
    sql_execute(sql, fetch_all=False)
    return {'status': 1}


def db_isAuthDataValid(data):
    sql='''
        SELECT user_id
        FROM authentications
        WHERE login='{login}' AND password='{password}';
    '''.format(**data)
    answer = sql_execute(sql, fetch_all=False)

    return bool(answer['user_id'])


def db_isProfileExists(data):
    sql = "SELECT count(login) FROM authentications "

    if type(data) == int:
        <vul/>sql += "<vul/>WHERE user_id='%d';</vul>" % data</vul>
    elif type(data) == dict:
        <vul/>sql += "WHERE login='%(login)s';" % data</vul>


    users = sql_execute(sql, fetch_all=False)['count']
    return bool(users)


def db_setLastVisit(ID):
    sql='''
        UPDATE users
        SET last_visit = NOW()
        <vul/><vul/>WHERE id='%d';
    ''' % ID</vul></vul>
    sql_execute(sql, fetch_all=False)


""" 
# Функция блокирует пользователя. 
По дефолту стоит True, поэтому аргумент status можно не отправлять. 
Если передать False, то разблокирует.
"""
def db_blockProfile(ID, status=True):
    sql='''
        UPDATE users
        <vul/>SET is_blocked='%s'
        WHERE id='%d';
    ''' % (status, ID)</vul>
    sql_execute(sql, fetch_all=False)



def db_getUserID(data):
    sql='''
        SELECT user_id
        FROM authentications
        <vul/>WHERE login='%(login)s';
    ''' % data</vul>
    user_id = sql_execute(sql, fetch_all=False)
    return user_id['user_id']




## PUBLIC METHODS
""" 
# Функция удаляет профиль пользователя. 
По дефолту стоит True, поэтому аргумент status можно не отправлять. 
Если передать False, то восстанавливает.
"""
def db_delProfile(ID, status=True):
    # TODO: Добавить запрос на удаление пользователя
    sql='''
        UPDATE users 
        <vul/>SET is_deleted='%s'
        WHERE id='%d';
    ''' % (status, ID)</vul>
    return sql_execute(sql, fetch_all=True)


def db_FullDelProfile(ID):
    # TODO: Добавить запрос на удаление пользователя
    sql='''
        DELETE FROM authentications
        WHERE user_id='%d';
        DELETE FROM users
        <vul/>WHERE id='%d';
    ''' % (ID, ID)</vul>
    sql_execute(sql, fetch_all=True)
    return {'status': 1}


def db_getProfileInfo(ID):
    sql='''
        SELECT first_name, second_name, id, last_visit, is_deleted, is_blocked
        FROM users
        WHERE id='%d';
    ''' % ID
    return sql_execute(sql, fetch_all=False)


def db_getProfilesInfo():
    sql='''
        SELECT first_name, second_name, id, last_visit, is_deleted, is_blocked
        FROM users;
    '''
    return sql_execute(sql, fetch_all=True)


def db_updateProfileInfo(ID, data):
    rows = []
    for key in data:
        if not key in ('first_name', 'second_name'):
            return {'status': 0, 'message': 'Неизвестное поле. Менять можно только first_name/second_name'}

        if data[key]:
            sql='''
                SELECT first_name, second_name
                FROM users
                <vul/>WHERE id='%d'
            ''' % ID</vul>
            answer = sql_execute(sql, fetch_all=False)

            if data[key] == answer[key]: # Если введённое и из БД поля эквиваленты, то выкидываем ошибку.
                rows.append(key)
                continue

            sql = '''
                UPDATE users
                <vul/>SET %s='%s' 
                WHERE id='%d';
            ''' % (key, data[key], ID)</vul>
            sql_execute(sql, fetch_all=False)

    if not len(rows):
        return {'status': 1}
    elif len(rows) >= 1:
        return {'status': 1, 'message': 'Эквивалентное поле {} не было изменено'.format(rows)}
    else:
        return {'status': 0, 'message': 'Эквивалентные поля {} не были изменены'.format(rows)}

