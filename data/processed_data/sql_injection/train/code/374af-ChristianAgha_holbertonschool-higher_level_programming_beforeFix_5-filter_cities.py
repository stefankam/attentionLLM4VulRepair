#!/usr/bin/python3
"""
takes in the name of a state as an argument and lists all cities of that state
"""
import MySQLdb
from sys import argv

if __name__ == "__main__":

    <vul/>city_list = []
    db = MySQLdb.Connect(host="localhost", port=3306, user=argv[1],
                         passwd=argv[2], db=argv[3])</vul>

    <vul/>cur = db.cursor()
    cur.execute("""
    SELECT cities.id, cities.name, states.name
    FROM cities
    JOIN states ON cities.state_id=states.id
    ORDER BY cities.id ASC""")</vul>

    <vul/>cities = cur.fetchall()
    for city in cities:
        if city[2] == argv[4]:
            city_list.append(city[1])</vul>

    <vul/>print(", ".join(city_list))</vul>

    <vul/>cur.close()
    db.close()</vul>
