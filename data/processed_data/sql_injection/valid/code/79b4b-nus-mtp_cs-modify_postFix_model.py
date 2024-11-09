'''
    model.py
    Handles queries to the database
'''

import hashlib
import components.database_adapter # database_adaptor.py handles the connection to database
import psycopg2

## Connects to the postgres database
CONNECTION = components.database_adapter.connect_db()
DB_CURSOR = CONNECTION.cursor()


def get_all_modules():
    '''
        Get the module code, name, description, and MCs of all modules
    '''
    sql_command = "SELECT * FROM module ORDER BY code"
    DB_CURSOR.execute(sql_command)
    return DB_CURSOR.fetchall()


def get_module(code):
    '''
        Get the module code, name, description and MCs of a single module
    '''
    sql_command = "SELECT * FROM module WHERE code=%s"
    <fix/>DB_CURSOR.execute(sql_command, (code,))</fix>
    return DB_CURSOR.fetchone()


def get_all_fixed_mounted_modules():
    '''
        Get the module code, name, AY/Sem and quota of all fixed mounted modules
    '''
    sql_command = "SELECT m2.moduleCode, m1.name, m2.acadYearAndSem, m2.quota " +\
                    "FROM module m1, moduleMounted m2 WHERE m2.moduleCode = m1.code " +\
                    "ORDER BY m2.moduleCode, m2.acadYearAndSem"
    DB_CURSOR.execute(sql_command)
    return DB_CURSOR.fetchall()


def get_all_tenta_mounted_modules():
    '''
        Get the module code, name, AY/Sem and quota of all tentative mounted modules
    '''
    sql_command = "SELECT m2.moduleCode, m1.name, m2.acadYearAndSem, m2.quota " +\
                    "FROM module m1, moduleMountTentative m2 WHERE m2.moduleCode = m1.code " +\
                    "ORDER BY m2.moduleCode, m2.acadYearAndSem"
    DB_CURSOR.execute(sql_command)
    return DB_CURSOR.fetchall()


def get_all_tenta_mounted_modules_of_selected_ay(selected_ay):
    '''
        Get the module code, name, AY/Sem and quota of all tenta mounted mods of a selected AY
    '''
    sql_command = "SELECT m2.moduleCode, m1.name, m2.acadYearAndSem, m2.quota " +\
                  "FROM module m1, moduleMountTentative m2 WHERE m2.moduleCode = m1.code " +\
                  "AND M2.acadYearAndSem LIKE %s" + \
                  "ORDER BY m2.moduleCode, m2.acadYearAndSem"
    processed_ay = selected_ay + "%"

    DB_CURSOR.execute(sql_command, (processed_ay,))
    return DB_CURSOR.fetchall()


def get_first_fixed_mounting():
    '''
        Get the first mounting from the fixed mounting table
        This is used for reading the current AY
    '''
    sql_command = "SELECT acadYearAndSem FROM moduleMounted LIMIT(1)"
    DB_CURSOR.execute(sql_command)
    return DB_CURSOR.fetchone()


def get_all_fixed_ay_sems():
    '''
        Get all the distinct AY/Sem in the fixed mounting table
    '''
    sql_command = "SELECT DISTINCT acadYearAndSem FROM moduleMounted " +\
                  "ORDER BY acadYearAndSem ASC"
    DB_CURSOR.execute(sql_command)
    return DB_CURSOR.fetchall()


def get_all_tenta_ay_sems():
    '''
        Get all the distinct AY/Sem in the tentative mounting table
    '''
    sql_command = "SELECT DISTINCT acadYearAndSem FROM moduleMountTentative " +\
                  "ORDER BY acadYearAndSem ASC"
    DB_CURSOR.execute(sql_command)
    return DB_CURSOR.fetchall()


def get_fixed_mounting_and_quota(code):
    '''
        Get the fixed AY/Sem and quota of a mounted module
    '''
    sql_command = "SELECT acadYearAndSem, quota FROM moduleMounted " +\
                  "WHERE moduleCode=%s ORDER BY acadYearAndSem ASC"
    DB_CURSOR.execute(sql_command, (code, ))
    return DB_CURSOR.fetchall()


def get_tenta_mounting_and_quota(code):
    '''
        Get the tentative AY/Sem and quota of a mounted module
    '''
    sql_command = "SELECT acadYearAndSem, quota FROM moduleMountTentative " +\
                  "WHERE moduleCode=%s ORDER BY acadYearAndSem ASC"
    DB_CURSOR.execute(sql_command, (code, ))
    return DB_CURSOR.fetchall()


def get_quota_of_target_fixed_ay_sem(code, ay_sem):
    '''
        Get the quota of a mod in a target fixed AY/Sem (if any)
    '''
    sql_command = "SELECT quota FROM moduleMounted " +\
                  "WHERE moduleCode=%s AND acadYearAndSem=%s "
    DB_CURSOR.execute(sql_command, (code, ay_sem))
    return DB_CURSOR.fetchall()


def get_quota_of_target_tenta_ay_sem(code, ay_sem):
    '''
        Get the quota of a mod in a target tentative AY/Sem (if any)
    '''
    sql_command = "SELECT quota FROM moduleMountTentative " +\
                  "WHERE moduleCode=%s AND acadYearAndSem=%s "
    DB_CURSOR.execute(sql_command, (code, ay_sem))
    return DB_CURSOR.fetchall()


def get_number_students_planning(code):
    '''
        Get the number of students planning to take a mounted module
    '''
    sql_command = "SELECT COUNT(*), acadYearAndSem FROM studentPlans WHERE " +\
                    "moduleCode=%s GROUP BY acadYearAndSem ORDER BY acadYearAndSem"
    DB_CURSOR.execute(sql_command, (code, ))
    return DB_CURSOR.fetchall()


def add_module(code, name, description, module_credits, status):
    '''
        Insert a module into the module table.
        Returns true if successful, false if duplicate primary key detected
    '''
    sql_command = "INSERT INTO module VALUES (%s,%s,%s,%s,%s)"
    try:
        DB_CURSOR.execute(sql_command, (code, name, description, module_credits, status))
        CONNECTION.commit()
    except psycopg2.IntegrityError:        # duplicate key error
        CONNECTION.rollback()
        return False
    return True


def update_module(code, name, description, module_credits):
    '''
        Update a module with edited info
    '''
    sql_command = "UPDATE module SET name=%s, description=%s, mc=%s " +\
                  "WHERE code=%s"
    try:
        DB_CURSOR.execute(sql_command, (name, description, module_credits, code))
        CONNECTION.commit()
    except psycopg2.Error:
        CONNECTION.rollback()
        return False
    return True


def flag_module_as_removed(code):
    '''
        Change the status of a module to 'To Be Removed'
    '''
    sql_command = "UPDATE module SET status='To Be Removed' WHERE code=%s"
    DB_CURSOR.execute(sql_command, (code, ))
    CONNECTION.commit()


def flag_module_as_active(code):
    '''
        Change the status of a module to 'Active'
    '''
    sql_command = "UPDATE module SET status='Active' WHERE code=%s"
    DB_CURSOR.execute(sql_command, (code, ))
    CONNECTION.commit()


def delete_module(code):
    '''
        Delete a module from the module table
    '''
    # Delete the foreign key reference first.
    sql_command = "DELETE FROM modulemounted WHERE modulecode=%s"
    DB_CURSOR.execute(sql_command, (code,))

    # Perform the normal delete.
    sql_command = "DELETE FROM module WHERE code=%s"
    DB_CURSOR.execute(sql_command, (code,))
    CONNECTION.commit()


def get_oversub_mod():
    '''
        Retrieves a list of modules which are oversubscribed.
        Returns module, AY/Sem, quota, number students interested
        i.e. has more students interested than the quota
    '''
    list_of_oversub_with_info = []
    list_all_mod_info = get_all_modules()

    for module_info in list_all_mod_info:
        mod_code = module_info[0]

        aysem_quota_fixed_list = get_fixed_mounting_and_quota(mod_code)
        aysem_quota_tenta_list = get_tenta_mounting_and_quota(mod_code)
        aysem_quota_merged_list = aysem_quota_fixed_list + \
                                aysem_quota_tenta_list

        num_student_plan_aysem_list = get_number_students_planning(mod_code)
        for num_plan_aysem_pair in num_student_plan_aysem_list:
            num_student_planning = num_plan_aysem_pair[0]
            ay_sem = num_plan_aysem_pair[1]
            real_quota = get_quota_in_aysem(ay_sem, aysem_quota_merged_list)

            # ensures that quota will be a number which is not None
            if real_quota is None:
                quota = 0
                real_quota = '?'
            else:
                quota = real_quota

            if num_student_planning > quota:
                oversub_info = (mod_code, ay_sem, real_quota, num_student_planning)
                list_of_oversub_with_info.append(oversub_info)

    return list_of_oversub_with_info


def get_quota_in_aysem(ay_sem, aysem_quota_merged_list):
    '''
        This is a helper function.
        Retrieves the correct quota from ay_sem listed inside
        aysem_quota_merged_list parameter.
    '''
    for aysem_quota_pair in aysem_quota_merged_list:
        aysem_in_pair = aysem_quota_pair[0]
        if ay_sem == aysem_in_pair:
            quota_in_pair = aysem_quota_pair[1]

            return quota_in_pair

    return None # quota not found in list


def add_admin(username, salt, hashed_pass):
    '''
        Register an admin into the database.
        Note: to change last argument to false once
        activation done
    '''
    sql_command = "INSERT INTO admin VALUES (%s, %s, %s, FALSE, TRUE)"
    DB_CURSOR.execute(sql_command, (username, salt, hashed_pass))
    CONNECTION.commit()


def is_userid_taken(userid):
    '''
        Retrieves all account ids for testing if a user id supplied
        during account creation
    '''
    sql_command = "SELECT staffid FROM admin WHERE staffID=%s"
    DB_CURSOR.execute(sql_command, (userid,))

    result = DB_CURSOR.fetchall()
    return len(result) != 0


def delete_admin(username):
    '''
        Delete an admin from the database.
    '''
    # Delete the foreign key references first.
    sql_command = "DELETE FROM starred WHERE staffID=%s"
    DB_CURSOR.execute(sql_command, (username,))

    sql_command = "DELETE FROM admin WHERE staffID=%s"
    DB_CURSOR.execute(sql_command, (username,))
    CONNECTION.commit()


def validate_admin(username, unhashed_pass):
    '''
        Check if a provided admin-password pair is valid.
    '''
    sql_command = "SELECT salt, password FROM admin WHERE staffID=%s"
    DB_CURSOR.execute(sql_command, (username,))
    admin = DB_CURSOR.fetchall()
    if not admin:
        return False
    else:
        hashed_pass = hashlib.sha512(unhashed_pass + admin[0][0]).hexdigest()
        is_valid = (admin[0][1] == hashed_pass)
        return is_valid


def add_fixed_mounting(code, ay_sem, quota):
    '''
        Insert a new mounting into fixed mounting table
    '''
    try:
        sql_command = "INSERT INTO modulemounted VALUES (%s,%s,%s)"
        DB_CURSOR.execute(sql_command, (code, ay_sem, quota))
        CONNECTION.commit()
    except psycopg2.IntegrityError:        # duplicate key error
        CONNECTION.rollback()
        return False
    return True


def delete_fixed_mounting(code, ay_sem):
    '''
        Delete a mounting from the fixed mounting table
    '''
    sql_command = "DELETE FROM modulemounted WHERE moduleCode=%s AND acadYearAndSem=%s"
    DB_CURSOR.execute(sql_command, (code, ay_sem))
    CONNECTION.commit()


def add_tenta_mounting(code, ay_sem, quota):
    '''
        Insert a new mounting into tentative mounting table
    '''
    try:
        sql_command = "INSERT INTO moduleMountTentative VALUES (%s,%s,%s)"
        DB_CURSOR.execute(sql_command, (code, ay_sem, quota))
        CONNECTION.commit()
    except psycopg2.IntegrityError:        # duplicate key error
        CONNECTION.rollback()
        return False
    return True


def update_quota(code, ay_sem, quota):
    '''
        Update the quota of a module in a target tentative AY/Sem
    '''
    sql_command = "UPDATE moduleMountTentative SET quota=%s " +\
                  "WHERE moduleCode=%s AND acadYearAndSem=%s"
    try:
        DB_CURSOR.execute(sql_command, (quota, code, ay_sem))
        CONNECTION.commit()
    except psycopg2.Error:
        CONNECTION.rollback()
        return False
    return True


def delete_tenta_mounting(code, ay_sem):
    '''
        Delete a mounting from the tentative mounting table
    '''
    sql_command = "DELETE FROM moduleMountTentative WHERE moduleCode=%s AND acadYearAndSem=%s"
    try:
        DB_CURSOR.execute(sql_command, (code, ay_sem))
        CONNECTION.commit()
    except psycopg2.Error:
        CONNECTION.rollback()
        return False
    return True


def get_num_students_by_yr_study():
    '''
        Retrieves the number of students at each year of study as a table
        Each row will contain (year, number of students) pair.
        e.g. [(1, 4), (2, 3)] means four year 1 students
        and two year 3 students
    '''
    INDEX_FIRST_ELEM = 0

    sql_command = "SELECT year, COUNT(*) FROM student GROUP BY year" + \
        " ORDER BY year"
    DB_CURSOR.execute(sql_command)

    table_with_non_zero_students = DB_CURSOR.fetchall()
    final_table = append_missing_year_of_study(table_with_non_zero_students)

    # Sort the table based on year
    final_table.sort(key=lambda row: row[INDEX_FIRST_ELEM])

    return final_table


def append_missing_year_of_study(initial_table):
    '''
        Helper function to append missing years of study to the
        given initial table.
        initial_table given in lists of (year, number of students)
        pair.
        e.g. If year 5 is missing from table, appends (5,0) to table
        and returns the table
    '''
    MAX_POSSIBLE_YEAR = 6
    for index in range(0, MAX_POSSIBLE_YEAR):
        year = index + 1
        year_exists_in_table = False

        for year_count_pair in initial_table:
            req_year = year_count_pair[0]
            if req_year == year:
                year_exists_in_table = True
                break

        if not year_exists_in_table:
            initial_table.append((year, 0))

    return initial_table


def get_num_students_by_focus_area_non_zero():
    '''
        Retrieves the number of students for each focus area as a table,
        if no student is taking that focus area, that row will not be
        returned.
        Each row will contain (focus area, number of students) pair.
        See: get_num_students_by_focus_areas() for more details.
    '''
    sql_command = "SELECT f.name, COUNT(*) FROM focusarea f, takesfocusarea t" + \
        " WHERE f.name = t.focusarea1 OR f.name = t.focusarea2 GROUP BY f.name"
    DB_CURSOR.execute(sql_command)

    return DB_CURSOR.fetchall()


def get_focus_areas_with_no_students_taking():
    '''
        Retrieves a list of focus areas with no students taking.
    '''
    sql_command = "SELECT f2.name FROM focusarea f2 WHERE NOT EXISTS(" + \
        "SELECT f.name FROM focusarea f, takesfocusarea t " + \
        "WHERE (f.name = t.focusarea1 OR f.name = t.focusarea2) " + \
        "AND f2.name = f.name GROUP BY f.name)"
    DB_CURSOR.execute(sql_command)

    return DB_CURSOR.fetchall()


def get_number_students_without_focus_area():
    '''
        Retrieves the number of students who have not indicated their focus
        area.
    '''
    sql_command = "SELECT COUNT(*) FROM takesfocusarea WHERE " + \
        "focusarea1 IS NULL AND focusarea2 IS NULL"
    DB_CURSOR.execute(sql_command)

    return DB_CURSOR.fetchone()


def get_num_students_by_focus_areas():
    '''
        Retrieves the number of students for each focus area as a table
        Each row will contain (focus area, number of students) pair
        e.g. [(AI, 4), (Database, 3)] means four students taking AI as
        focus area and three students taking database as focus area.
        Note: A student taking double focus on AI and Database will be
        reflected once for AI and once for database (i.e. double counting)
    '''
    INDEX_FIRST_ELEM = 0

    table_with_non_zero_students = get_num_students_by_focus_area_non_zero()
    table_with_zero_students = get_focus_areas_with_no_students_taking()

    temp_table = table_with_non_zero_students

    # Loops through all focus areas with no students taking and add them to
    # the table with (focus area, number of students) pair.
    for focus_area_name in table_with_zero_students:
        temp_table.append((focus_area_name[INDEX_FIRST_ELEM], 0))

    # Sort the table based on focus area
    temp_table.sort(key=lambda row: row[INDEX_FIRST_ELEM])

    # Build the final table with info of students without focus area.
    num_students_without_focus = \
    get_number_students_without_focus_area()[INDEX_FIRST_ELEM]

    temp_table.insert(INDEX_FIRST_ELEM,
                      ("Have Not Indicated", num_students_without_focus))
    final_table = temp_table

    return final_table

def get_mod_taken_together_with(code):
    '''
        Retrieves the list of modules taken together with the specified
        module code in the same semester.

        Returns a table of lists (up to 10 top results). Each list contains
        (specified code, module code of mod taken together, aySem, number of students)

        e.g. [(CS1010, CS1231, AY 16/17 Sem 1, 5)] means there are 5 students
        taking CS1010 and CS1231 together in AY 16/17 Sem 1.
    '''
    NUM_TOP_RESULTS_TO_RETURN = 10

    sql_command = "SELECT sp1.moduleCode, sp2.moduleCode, sp1.acadYearAndSem, COUNT(*) " + \
                "FROM studentPlans sp1, studentPlans sp2 " + \
                <fix/>"WHERE sp1.moduleCode = %s AND " + \</fix>
                "sp2.moduleCode <> sp1.moduleCode AND " + \
                "sp1.studentId = sp2.studentId AND " + \
                "sp1.acadYearAndSem = sp2.acadYearAndSem " + \
                "GROUP BY sp1.moduleCode, sp2.moduleCode, sp1.acadYearAndSem " + \
                "ORDER BY COUNT(*) DESC"

    DB_CURSOR.execute(sql_command, (code,))

    return DB_CURSOR.fetchmany(NUM_TOP_RESULTS_TO_RETURN)

def get_all_mods_taken_together():
    '''
        Retrieves the list of all modules taken together in the same semester.

        Returns a table of lists. Each list contains
        (module code 1, module code 2, aySem, number of students)
        where module code 1 and module code 2 are the 2 mods taken together
        in the same semester.

        e.g. [(CS1010, CS1231, AY 16/17 Sem 1, 5)] means there are 5 students
        taking CS1010 and CS1231 together in AY 16/17 Sem 1.
    '''

    sql_command = "SELECT sp1.moduleCode, sp2.moduleCode, sp1.acadYearAndSem, COUNT(*) " + \
                "FROM studentPlans sp1, studentPlans sp2 " + \
                "WHERE sp1.moduleCode < sp2.moduleCode AND " + \
                "sp1.studentId = sp2.studentId AND " + \
                "sp1.acadYearAndSem = sp2.acadYearAndSem " + \
                "GROUP BY sp1.moduleCode, sp2.moduleCode, sp1.acadYearAndSem " + \
                "ORDER BY COUNT(*) DESC"

    DB_CURSOR.execute(sql_command)

    return DB_CURSOR.fetchall()
