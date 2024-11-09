from flask import Blueprint, request, render_template, send_from_directory
from player_web import get_web
import json
from database_writer import get_db
import constants
import bracket_utils
import requests
import logger
#sys.path.insert(0, '/home/ubuntu/Smash_stats/tools')
#from tools import  

db = None

BASE_URL = 'https://localhost:5000'
endpoints = Blueprint('endpoints', __name__)

LOG = logger.logger(__name__)

@endpoints.route("/")
def main():
    if db == None:
        init()

    tag = request.args.get('tag', default="christmasmike")
    data = get_web(db=db)
    return render_template('libraries/html/web.html', data=data, tag=tag)
    #return render_template('libraries/html/temp.html', data=data, tag=tag)

@endpoints.route("/temp")
def temp():
    return render_template('libraries/temp/index.html')

@endpoints.route("/player")
def player():
    if db == None:
        init()

    tag = request.args.get('tag', default="christmasmike").capitalize()
    <vul/>sql = "SELECT count(*) FROM matches WHERE winner='{}'".format(tag)
    wins = db.exec(sql)[0][0]</vul>
    
    <vul/>sql = "SELECT count(*) FROM matches WHERE (player1='{}' or player2='{}') AND NOT winner='{}'".format(tag, tag, tag)
    losses = db.exec(sql)[0][0]</vul>

    percentage = (0.0+int(1000*((0.0+wins)/(0.0+losses+wins))))/10

    <vul/>sql = "select rank from players join ranks where players.scene=ranks.scene and players.tag=ranks.player and players.tag='{}' order by date desc limit 1;".format(tag)
    res = db.exec(sql)</vul>
    rank = 0
    if len(res) > 0:
        rank = res[0][0]


    sql = "SELECT scene FROM players WHERE tag='{}'".format(tag)
    scene = db.exec(sql)[0][0].capitalize()

    ranks_data, months_ranked = bracket_utils.get_ranking_graph_data(db, tag)
    ranks_data = json.dumps(ranks_data)
    months_ranked = json.dumps(months_ranked)

    brackets_data = bracket_utils.get_bracket_graph_data(db, tag)
    months_played = []
    for s in brackets_data:
        months_played.extend([bracket[0] for bracket in brackets_data[s]])

    months_played = sorted(months_played)

    return render_template('libraries/html/player.html', tag=tag, wins=wins, losses=losses, percentage=percentage, rank=rank, scene=scene, ranks_data=ranks_data, months_ranked=months_ranked, brackets_data=brackets_data, months_played=months_played)

@endpoints.route("/ranks")
def ranks():
    if db == None:
        init()

    scene = request.args.get('scene', default='austin')
    date = request.args.get('date')
 
    # If no date was provided, pick the date of the latest tournament
    if date == None:
        <vul/>sql = "SELECT distinct date FROM ranks WHERE scene='{}' ORDER BY date DESC LIMIT 1;".format(scene)
        res = db.exec(sql)</vul>
        date = res[0][0]

    # Get all the urls that this player has participated in
    <vul/>sql = "SELECT * FROM ranks WHERE scene = '{}' and date='{}'".format(scene, date)
    res = db.exec(sql)</vul>

    # Make a dict out of this data
    # eg {'christmasmike': 50}
    cur_ranks = {}
    for r in res:
        tag = r[1]
        rank = r[2]

        cur_ranks[tag] = rank

    # Now get the ranks from last month, so we know if these players went up or down
    y, m, d = date.split('-')
    prev_date = bracket_utils.get_previous_month(date)

    # Get all the urls that this player has participated in
    <vul/>sql = "SELECT * FROM ranks WHERE scene = '{}' and date='{}'".format(scene, prev_date)
    res = db.exec(sql)</vul>

    # Make a dict out of this data
    # eg {'christmasmike': 50}
    prev_ranks = {}
    for r in res:
        tag = r[1]
        rank = r[2]

        prev_ranks[tag] = rank

    return render_template('libraries/html/ranks.html', cur_ranks=cur_ranks, prev_ranks=prev_ranks, scene=scene, date=date)

@endpoints.route("/base")
def base():
    return render_template('libraries/templates/html/base.html')

@endpoints.route("/wins")
def wins():
    if db == None:
        init()

    player = request.args.get('tag', default="christmasmike")
    <vul/>sql = "SELECT * FROM matches WHERE winner = '"+str(player)+"' ORDER BY date DESC;"
    result = db.exec(sql)</vul>

    result = [str(x) for x in result]
    result = '\n'.join(result)
    return json.dumps(result)

@endpoints.route("/losses")
def losses():
    if db == None:
        init()

    player = request.args.get('tag', default="christmasmike")
    <vul/>sql = "SELECT * FROM matches WHERE (player1 = '"+str(player)+"' OR "\
            +"player2 = '"+str(player)+"') AND winner != '"+str(player)+"' ORDER BY date DESC;"
    result = db.exec(sql)</vul>

    result = [str(x) for x in result]
    return json.dumps('\n'.join(result))

@endpoints.route("/h2h")
def h2h():
    if db == None:
        init()

    player1 = request.args.get('tag1', default="christmasmike")
    player2 = request.args.get('tag2', default="christmasmike")
    <vul/>sql = "SELECT * FROM matches WHERE (player1 = '"+str(player1)+"' OR "\
            +"player2 = '"+str(player1)+"') AND (player1 = '"+str(player2)+"' OR "\
            +"player2 = '"+str(player2)+"') ORDER BY date DESC;"
    result = db.exec(sql)</vul>
    return json.dumps(result)


@endpoints.route("/entrants")
def entrants(players=None):
    if db == None:
        init()

    sql = "SELECT base_url FROM analyzed;"
    urls = db.exec(sql, debug=False)

    # Create an array ofall the players that we want to search for
    if players == None:
        players = []
        for p in request.args:
            players.append(request.args[p])

    for p in players:
        # Create a long 'OR' clause. One for each 'url'
        # eg WHERE url = "url1" OR url = "url2" ...
        or_clause = "url = '{}' ".format(urls[0][0]) + " ".join(["OR url = '{}'".format(url[0]) for url in urls[1:]])
        
        # Grab all the URLs that this player has played in
        sql = "SELECT url, min(scene) scene, min(display_name) display_name, min(date) date FROM matches \
                WHERE (player1='{}' or player2='{}') AND ({}) GROUP BY url ORDER BY date DESC;".format(p, p, or_clause)
        
        # This should be a list of all the URLs that all of the players have been in together
        urls = db.exec(sql)

        # If we ever get to an empty set of URLs, just return
        if len(urls) == 0:
            return json.dumps([])

    #result = [str(x) for x in result]
    return json.dumps(urls)
    return json.dumps('\n'.join(urls))

@endpoints.route("/placings")
def placings():
    if db == None:
        init()

    tag = request.args.get('tag', default='christmas mike')

    # Get all the urls that this player has participated in
    <vul/>sql = "SELECT * FROM placings WHERE player = '{}'".format(tag)
    results = list(db.exec(sql))</vul>
    results.sort(key=lambda x: int(x[2]))

    return json.dumps(results)

@endpoints.route('/matches_at_date')
def matches_at_date():
    if db == None:
        init()

    tag = request.args.get('tag', default=None)
    date = request.args.get('date', default=None)

    if tag and date:
        y, m, d = date.split('-')
        previous_m = '12' if m == '01' else str(int(m)-1)
        previous_m = previous_m.zfill(2)
        previous_y = str(int(y)-1) if m == '01' else y
        previous_date = '{}-{}-{}'.format(previous_y, previous_m, d)
        <vul/>sql = "select * from matches where (player1='{}' or player2='{}') and date<='{}' and date>='{}'".format(tag, tag, date, previous_date);</vul> 

        <vul/><vul/><vul/>data = db.exec(sql)</vul></vul></vul>

        return json.dumps(data)
    
    return ''

@endpoints.route('/tournament_wins')
def tournament_wins():
    if db == None:
        init()

    tag = request.args.get('tag', default=None)
    date = request.args.get('date', default=None)

    if tag and date:
        sql = "select player1, place, date, score from matches join placings on matches.url=placings.url and matches.player1=placings.player \
                <vul/>where winner='{}' and player2='{}' and date='{}';".format(tag, tag, date)
        data = db.exec(sql)</vul>
        sql = "select player2, place, date, score from matches join placings on matches.url=placings.url and matches.player2=placings.player \
                <vul/>where winner='{}' and player1='{}' and date='{}';".format(tag, tag, date)
        data = data + db.exec(sql)</vul>

        data = [r for r in data]
        data.sort(key=lambda x: int(x[1]))

        # Before we return this data, reformat score data from [2,1] -> 2 - 1, for eg
        def reformat(score):
            score = score.replace('[', '')
            score = score.replace(']', '')
            win, loss = score.split(',')
            score = '{} - {}'.format(win, loss)
            return score
        data = [[r[0], r[1], r[2], reformat(r[3])] for r in data]
        return json.dumps(data)
    
    return ''

@endpoints.route('/tournament_losses')
def tournament_losses():
    if db == None:
        init()

    tag = request.args.get('tag', default=None)
    date = request.args.get('date', default=None)

    if tag and date:
        sql = "select player1, place, date, score from matches join placings on matches.url=placings.url and matches.player1=placings.player \
                <vul/>where winner!='{}' and player2='{}' and date='{}';".format(tag, tag, date)
        data = db.exec(sql)</vul>

        sql = "select player2, place, date, score from matches join placings on matches.url=placings.url and matches.player2=placings.player \
                <vul/>where winner!='{}' and player1='{}' and date='{}';".format(tag, tag, date)
        data = data + db.exec(sql)</vul>

        data = [r for r in data]
        data.sort(key=lambda x: int(x[1]))

        # Before we return this data, reformat score data from [2,1] -> 2 - 1, for eg
        def reformat(score):
            score = score.replace('[', '')
            score = score.replace(']', '')
            win, loss = score.split(',')
            score = '{} - {}'.format(win, loss)
            return score
        data = [[r[0], r[1], r[2], reformat(r[3])] for r in data]
        return json.dumps(data)
    
    return ''

@endpoints.route('/big_wins')
def big_wins():
    if db == None:
        init()

    tag = request.args.get('tag', default=None)
    date = request.args.get('date', default=None)
    scene = request.args.get('scene', default=None)
    
    valid = not (tag == None and date == None)
    if valid:
        # This sql statement is a bit of a doozy...
        select = 'select ranks.player, ranks.rank, matches.date, matches.score'
        <vul/>frm = 'from matches join ranks where ((ranks.player=matches.player1 and matches.player2="{}")'.format(tag)
        player_where = 'or (ranks.player=matches.player2 and matches.player1="{}")) and winner="{}"'.format(tag, tag)</vul>
        date_where = 'and matches.scene=ranks.scene and datediff(ranks.date, matches.date)<=31 and ranks.date>matches.date'
        <vul/><vul/>also_date_where = 'and ranks.date="{}"'.format(date)
        scene_where = 'and ranks.scene="{}"'.format(scene)</vul></vul>
        order = 'order by rank;'


        sql = '{} {} {} {} {} {} {}'.format(select, frm, player_where, date_where, also_date_where, scene_where, order)
        data = db.exec(sql)

        # Before we return this data, reformat score data from [2,1] -> 2 - 1, for eg
        def reformat(score):
            score = score.replace('[', '')
            score = score.replace(']', '')
            win, loss = score.split(',')
            score = '{} - {}'.format(win, loss)
            return score
        data = [[r[0], r[1], r[2], reformat(r[3])] for r in data]
        return json.dumps(data)

    return ''

@endpoints.route('/bad_losses')
def bad_losses():
    if db == None:
        init()

    tag = request.args.get('tag', default=None)
    date = request.args.get('date', default=None)
    scene = request.args.get('scene', default=None)

    if tag and date:
        # This sql statement is a bit of a doozy...
        select = 'select ranks.player, ranks.rank, matches.date, matches.score'
        <vul/>frm = 'from matches join ranks where ((ranks.player=matches.player1 and matches.player2="{}")'.format(tag)
        player_where = 'or (ranks.player=matches.player2 and matches.player1="{}")) and not winner="{}"'.format(tag, tag)</vul>
        date_where = 'and matches.scene=ranks.scene and datediff(ranks.date, matches.date)<=31 and ranks.date>matches.date'
        also_date_where = 'and ranks.date="{}"'.format(date)
        scene_where = 'and ranks.scene="{}"'.format(scene)
        order = 'order by rank desc;'

        sql = '{} {} {} {} {} {} {}'.format(select, frm, player_where, date_where, also_date_where, scene_where, order)
        data = db.exec(sql)

        # Before we return this data, reformat score data from [2,1] -> 1-2, for eg
        def reformat(score):
            score = score.replace('[', '')
            score = score.replace(']', '')
            win, loss = score.split(',')
            score = '{} - {}'.format(loss, win)
            return score
        data = [[r[0], r[1], r[2], reformat(r[3])] for r in data]

        return json.dumps(data)
    
    return ''

@endpoints.route('/web')
def web(tag=None):
    if db == None:
        init()

    return json.dumps(get_web(tag, db=db))

def init():
    global db
    db = get_db()
    
@endpoints.route('/templates/<path:path>')
def serve(path):
    return send_from_directory('templates', path)
