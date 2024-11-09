#!/usr/bin/env python
# 
# tournament.py -- implementation of a Swiss-system tournament
#

import psycopg2
import bleach


def connect():
    """Connect to the PostgreSQL database.  Returns a database connection."""
    return psycopg2.connect("dbname=tournament")


def execute(query, params=None):
    conn = connect()
    c = conn.cursor()
    if not params:
        c.execute(query)
    else:
        c.execute(query, params)
    conn.commit()
    conn.close()


def fetchone(query):
    conn = connect()
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchone()[0]
    conn.close()
    return result 


def fetchall(query):
    conn = connect()
    cur = conn.cursor()
    cur.execute(query)
    records = cur.fetchall()
    conn.close()
    return records


def deleteMatches():
    """Remove all the match records from the database."""
    execute("delete from Match")


def deletePlayers():
    """Remove all the player records from the database."""
    execute("delete from Player")


def countPlayers():
    """Returns the number of players currently registered."""
    return fetchone("select count(*) from Player")


def registerPlayer(name):
    """Adds a player to the tournament database.
  
    The database assigns a unique serial id number for the player.  (This
    should be handled by your SQL database schema, not in your Python code.)
  
    Args:
      name: the player's full name (need not be unique).
    """

    # prevent xss
    name = bleach.clean(name)

    execute("insert into Player(name) values(%s)", (name,))


def playerStandings():
    """Returns a list of the players and their win records, sorted by wins.

    The first entry in the list should be the player in first place, or a player
    tied for first place if there is currently a tie.

    Returns:
      A list of tuples, each of which contains (id, name, wins, matches):
        id: the player's unique id (assigned by the database)
        name: the player's full name (as registered)
        wins: the number of matches the player has won
        matches: the number of matches the player has played
    """
    return fetchall("select * from player_static_view order by wins")


def reportMatch(winner, loser):
    """Records the outcome of a single match between two players.

    Args:
      winner:  the id number of the player who won
      loser:  the id number of the player who lost
    """

    # prevent xss
    winner = bleach.clean(str(winner))
    loser = bleach.clean(str(loser))

    execute("insert into Match(winner, loser) values(%s, %s)", (winner, loser, ))
 
 
def swissPairings():
    """Returns a list of pairs of players for the next round of a match.
  
    Assuming that there are an even number of players registered, each player
    appears exactly once in the pairings. Each player is paired with another
    player with an equal or nearly-equal win record, that is, a player adjacent
    to him or her in the standings.
  
    Returns:
      A list of tuples, each of which contains (id1, name1, id2, name2)
        id1: the first player's unique id
        name1: the first player's name
        id2: the second player's unique id
        name2: the second player's name
    """
    # fetch data
    records = fetchall("select * from player_static_view order by wins desc")

    # split data into pairs
    count = 0
    length = len(records)
    pairs = []
    while count < length:
        pairs.append((
            records[count][0],
            records[count][1],
            records[count+1][0],
            records[count+1][1],
        ))
        count += 2
    return pairs



