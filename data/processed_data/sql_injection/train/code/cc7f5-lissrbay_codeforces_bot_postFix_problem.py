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
colors = ['red', 'green', 'tan', 'blue', 'purple', 'orange']

<fix/>def checking_request_tags(tag):</fix>
    list_of_current_tags = list()
    <fix/>for i in available_tags:
        if i in tag:
            list_of_current_tags.append(i)</fix>

    <fix/>if len(list_of_current_tags) == 0:
        return available_tags.copy()</fix>

    <fix/>return list_of_current_tags</fix>


<fix/>def find_intersection(tasks, tag, username):
    conn = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + '.db')
    conn2 = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + '\\cf.db')
    cursor = conn.cursor()
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT * FROM " + tag)
    a = list()
    problem_and_diff = cursor2.fetchone()
    while problem_and_diff != None:
        cursor.execute("SELECT * FROM result WHERE problem = ? AND diff = ?  AND NOT verdict = 'OK'", (problem_and_diff[0], problem_and_diff[1]))
        problem_and_diff_and_ok = cursor.fetchone()
        if problem_and_diff_and_ok != None and problem_and_diff_and_ok in tasks:
            a.append(problem_and_diff_and_ok)
        problem_and_diff = cursor2.fetchone()
    conn.close()
    conn2.close()
    return a</fix>


<fix/>def get_array_of_tasks(tags_array, tasks, username):
    for i in range(1, len(tags_array)):
        tasks = find_intersection(tasks, tags_array[i], username)
    return tasks


def checking_request_diff(tag):</fix>
    list_of_current_diff = list()
    for i in available_diff:
        if i in tag:
            list_of_current_diff.append(i)
    <fix/>if len(list_of_current_diff) == 0:
        return available_diff.copy()
    return list_of_current_diff</fix>


def get_unsolved_problem(tag, username):
    tasks = list()
    list_of_current_tags = checking_request_tags(tag)
    list_of_current_diff = checking_request_diff(tag)
    conn = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + '.db')
    conn2 = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + '\\cf.db')
    cursor = conn.cursor()
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT * FROM " + list_of_current_tags[0])
    problem_and_diff = cursor2.fetchone()
    while problem_and_diff != None:
        if problem_and_diff[1] in list_of_current_diff:
            <fix/>cursor.execute("SELECT * FROM result WHERE problem = ? AND diff = ? AND NOT verdict = 'OK'", (problem_and_diff[0], problem_and_diff[1]))</fix>
            problem_and_diff_and_ok = cursor.fetchone()
            if problem_and_diff_and_ok != None:
                tasks.append(problem_and_diff_and_ok)
        problem_and_diff = cursor2.fetchone()
    conn.close()
    conn2.close()
    <fix/>tasks = get_array_of_tasks(list_of_current_tags, tasks, username)</fix>
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

<fix/>def count_stats(username):</fix>
    conn = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + '.db')
    conn2 = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + '\\cf.db')
    cursor = conn.cursor()
    cursor2 = conn2.cursor()
    <fix/>list_tags_stats = list()
</fix>
    for i in available_tags:
        cursor2.execute("SELECT * FROM " + str(i))
        x = cursor2.fetchone()
        count = 0
        while x != None:
            <fix/>cursor.execute("SELECT * FROM result WHERE problem = ? AND diff = ? AND verdict = 'OK'", (x[0], x[1]))</fix>
            y = cursor.fetchone()
            if y != None:
                count += 1
            x = cursor2.fetchone()
        <fix/>list_tags_stats.append(Pair(count, i))</fix>
    conn.close()
    conn2.close()
    <fix/>return list_tags_stats

def create_stats_picture(username):
    data_for_plot = list()
    leg = list()
    list_tags_stats = count_stats(username)
    sum = 0
    for i in range(len(list_tags_stats)):
        sum += list_tags_stats[i].first
    for i in range(len(list_tags_stats)):
        if list_tags_stats[i].first / sum != 0:
            data_for_plot.append(list_tags_stats[i].first / sum)
            leg.append(list_tags_stats[i].second)</fix>

    fig1, ax1 = plt.subplots()
    <fix/>ax1.pie(data_for_plot,  autopct='%1.1f%%',</fix>
            shadow=True, startangle=90)
    ax1.axis('equal')
    ax1.legend(leg)
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)) + "\\users\\", username + '.png')
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + ".png")
    plt.close()
    return False


<fix/>def count_stats_for_second_plot(username):
    verdict = {"COMPILATION_ERROR": 0, "OK": 0, "TIME_LIMIT_EXCEEDED": 0, "WRONG_ANSWER": 0, "RUNTIME_ERROR": 0,
               "MEMORY_LIMIT_EXCEEDED": 0}</fix>
    conn = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + '.db')
    conn2 = sqlite3.connect(os.path.abspath(os.path.dirname(__file__)) + '\\cf.db')
    cursor = conn.cursor()
    cursor2 = conn2.cursor()
    count = 0
    for i in available_tags:
        cursor2.execute("SELECT * FROM " + str(i))
        x = cursor2.fetchone()
        while x != None:
            <fix/>cursor.execute("SELECT * FROM result WHERE problem = ? AND diff = ?", (x[0], x[1]))</fix>
            y = cursor.fetchone()
            if y != None:
                for j in verdict.keys():
                    if y[2] == j:
                        verdict[j] += 1
                        count += 1

            x = cursor2.fetchone()
    return verdict
    conn.close()
    conn2.close()


def create_text_stats(username):
    list_tags_stats = list()
    data_for_plot = list()
    verdict = count_stats_for_second_plot(username)
    for i in verdict.keys():
        <fix/>list_tags_stats.append(i)
        data_for_plot.append(verdict[i])</fix>
    fig1, ax1 = plt.subplots()
    <fix/>ax1.pie(data_for_plot, labels = data_for_plot, colors = colors,</fix>
            shadow=True, startangle=90)
    ax1.axis('equal')
    <fix/>ax1.legend(list_tags_stats)</fix>
    ax1.set_title('How many different verdict in last status of problem you have: ')
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)) + "\\users\\", username + '.png')
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(os.path.abspath(os.path.dirname(__file__)) + "\\users\\" + username + ".png")
    plt.close()
    <fix/>return True</fix>




