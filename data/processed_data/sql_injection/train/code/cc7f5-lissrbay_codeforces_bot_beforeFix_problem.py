#Ребята, не стоит вскрывать этот код.
#Вы молодые, хакеры, вам все легко. Это не то.
#Это не Stuxnet и даже не шпионские программы ЦРУ. Сюда лучше не лезть.
#Серьезно, любой из вас будет жалеть. Лучше закройте компилятор и забудьте что там писалось.
#Я вполне понимаю что данным сообщением вызову дополнительный интерес, но хочу сразу предостеречь пытливых - стоп.
#Остальные просто не найдут.

import sqlite3
import os
import random
import matplotlib.pyplot as plt

available_tags = ['math', "strings", "trees", "graphs", "dp", "greedy", "geometry", "combinatorics"]
available_diff = ['A', 'B', 'C', 'D', 'E', 'F']

<vul/>def get_unsolved_problem(tag, username):
    tasks = list()</vul>
    list_of_current_tags = list()
    <vul/>list_of_current_diff = list()</vul>

    <vul/>def find_intersection(tag):
        conn = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + '.db')
        conn2 = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + '\\cf.db')
        cursor = conn.cursor()
        cursor2 = conn2.cursor()
        cursor2.execute("SELECT * FROM " + tag)
        a = list()
        problem_and_diff = cursor2.fetchone()</vul>

        <vul/>while problem_and_diff != None:
            cursor.execute("SELECT * FROM result WHERE problem = '" + str(problem_and_diff[0]) + "' AND diff = '" + str(problem_and_diff[1]) + "' AND NOT verdict = 'OK'")
            problem_and_diff_and_ok = cursor.fetchone()
            if problem_and_diff_and_ok != None and problem_and_diff_and_ok in tasks:
                a.append(problem_and_diff_and_ok)
            problem_and_diff = cursor2.fetchone()</vul>

        <vul/>conn.close()
        conn2.close()
        return a</vul>


    <vul/>list_of_current_tags = list()
    for i in available_tags:
        if i in tag:
            list_of_current_tags.append(i)</vul>

    list_of_current_diff = list()
    for i in available_diff:
        if i in tag:
            list_of_current_diff.append(i)
    <vul/>f = False
    if len(list_of_current_tags) == 0 and list_of_current_diff !=0:
        list_of_current_tags = available_tags.copy()
        f = True
    if len(list_of_current_tags) == 0 and len(list_of_current_diff) == 0:
        return "Plz try again"
    if len(list_of_current_diff) == 0 and len(list_of_current_tags) != 0:
        list_of_current_diff = available_diff.copy()
        f = True</vul>


    conn = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + '.db')
    conn2 = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + '\\cf.db')
    cursor = conn.cursor()
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT * FROM " + list_of_current_tags[0])
    problem_and_diff = cursor2.fetchone()

    while problem_and_diff != None:
        if problem_and_diff[1] in list_of_current_diff:
            cursor.execute("SELECT * FROM result WHERE problem = '" + str(problem_and_diff[0]) + "' AND diff = '" + str(
                problem_and_diff[1]) + "' AND NOT verdict = 'OK'")
            problem_and_diff_and_ok = cursor.fetchone()
            if problem_and_diff_and_ok != None:
                tasks.append(problem_and_diff_and_ok)
        problem_and_diff = cursor2.fetchone()

    conn.close()
    conn2.close()
    if not f:
        for i in range(1, len(list_of_current_tags)):
            tasks = find_intersection(list_of_current_tags[i])

    random.seed()
    if len(tasks) > 0:
        ind1 = random.randint(0, len(tasks) - 1)
        s1 = str(tasks[ind1][0]) + '/' + tasks[ind1][1]
        tasks.pop(ind1)
        return 'http://codeforces.com/problemset/problem/' + s1
    else:
        return "You have solved all tasks with this tag, nice!"

def get_theory_from_tag(tag):
    if not tag in available_tags:
        return "Incorrect tag."

    base = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\theory.db")
    conn = base.cursor()
    conn.execute("select * from " + tag)
    x = conn.fetchone()
    s = ""
    while x != None:
        s += str(x[0]) + '\n'
        x = conn.fetchone()
    base.close()
    return s

class Pair():
    def __init__(self, first, second):
        self.first = first
        self.second = second

<vul/>def create_stats_picture(username):</vul>
    conn = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + '.db')
    conn2 = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + '\\cf.db')
    cursor = conn.cursor()
    cursor2 = conn2.cursor()
    <vul/>a = list()
    b = list()
    leg = list()
    sum = 0</vul>
    for i in available_tags:
        cursor2.execute("SELECT * FROM " + str(i))
        x = cursor2.fetchone()
        count = 0
        while x != None:
            <vul/>cursor.execute("SELECT * FROM result WHERE problem = '" + str(x[0]) + "' AND diff = '" + str(
                x[1]) + "' AND verdict = 'OK'")</vul>
            y = cursor.fetchone()
            if y != None:
                count += 1
            x = cursor2.fetchone()
        <vul/>a.append(Pair(count, i))
        sum += count
</vul>
    conn.close()
    conn2.close()
    <vul/>if sum == 0:
        return True
    for i in range(len(a)):
        if a[i].first / sum != 0:
            b.append(a[i].first / sum)
            leg.append(a[i].second)</vul>

    fig1, ax1 = plt.subplots()
    <vul/>ax1.pie(b,  autopct='%1.1f%%',</vul>
            shadow=True, startangle=90)
    ax1.axis('equal')
    ax1.legend(leg)
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)) + "\\users\\", username + '.png')
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + ".png")
    plt.close()
    return False


<vul/>def create_text_stats(username):
    verdict = {"COMPILATION_ERROR" : 0, "OK" : 0, "TIME_LIMIT_EXCEEDED" : 0, "WRONG_ANSWER" : 0, "RUNTIME_ERROR" : 0, "MEMORY_LIMIT_EXCEEDED" : 0}
    colors = ['red', 'green', 'tan', 'blue', 'purple', 'orange']</vul>
    conn = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + '.db')
    conn2 = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + '\\cf.db')
    cursor = conn.cursor()
    cursor2 = conn2.cursor()
    count = 0
    a = list()
    b = list()
    for i in available_tags:
        cursor2.execute("SELECT * FROM " + str(i))
        x = cursor2.fetchone()
        while x != None:
            cursor.execute("SELECT * FROM result WHERE problem = '" + str(x[0]) + "' AND diff = '" + str(x[1]) + "'")
            y = cursor.fetchone()
            if y != None:
                for j in verdict.keys():
                    if y[2] == j:
                        verdict[j] += 1
                        count += 1

            x = cursor2.fetchone()

    for i in verdict.keys():
        <vul/>a.append(i)
        b.append(verdict[i])</vul>
    fig1, ax1 = plt.subplots()
    <vul/>ax1.pie(b, labels = b, colors = colors,</vul>
            shadow=True, startangle=90)
    ax1.axis('equal')
    <vul/>ax1.legend(a)</vul>
    ax1.set_title('How many different verdict in last status of problem you have: ')
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)) + "\\users\\", username + '.png')
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + ".png")
    conn.close()
    conn2.close()
    plt.close()
    s = username + " has at least one submissions in " + str(count) + " problems"
    return s



