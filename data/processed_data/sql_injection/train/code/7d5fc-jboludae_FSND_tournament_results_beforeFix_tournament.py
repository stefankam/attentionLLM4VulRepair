#!/usr/bin/env python
# 
# tournament.py -- implementation of a Swiss-system tournament
#

import psycopg2
<vul/>import bleach</vul>


def connect():
    """Connect to the PostgreSQL database.  Returns a database connection."""
    <vul/>return psycopg2.connect("dbname=tournament")</vul>


def deleteMatches():
    """Remove all the match records from the database."""
    <vul/>conn = connect()
    c = conn.cursor()
    c.execute("delete from matches")
    conn.commit()
    conn.close()</vul>


def deletePlayers():
    """Remove all the player records from the database."""
    <vul/>conn = connect()
    c = conn.cursor()
    c.execute("delete from players")
    conn.commit()
    conn.close()</vul>


def countPlayers():
    """Returns the number of players currently registered."""
    <vul/>conn = connect()
    c = conn.cursor()
    c.execute("select count(*) from players")
    results = c.fetchone()
    conn.close()
    return int(results[0])</vul>


def registerPlayer(name):
    """Adds a player to the tournament database.
  
    The database assigns a unique serial id number for the player.  (This
    should be handled by your SQL database schema, not in your Python code.)
  
    Args:
      name: the player's full name (need not be unique).
    """
    <vul/>name = bleach.clean(name)
    conn = connect()
    c = conn.cursor()
    # -->(%s ,0 ,0)",(name,)<-- this syntax is important to ' are inserted safely
    c.execute("insert into players (name_player) values (%s)",(name,))
    conn.commit()
    conn.close()</vul>


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
    <vul/>conn = connect()
    c = conn.cursor()
    c.execute("select * from ranking order by count_wins desc")
    results = c.fetchall()
    conn.commit()
    conn.close()
    return results</vul>


def reportMatch(winner, loser):
    """Records the outcome of a single match between two players.

    Args:
      winner:  the id number of the player who won
      loser:  the id number of the player who lost
    """
    <vul/>conn = connect()
    c = conn.cursor()
    # Insert match into matches table
    c.execute("insert into matches (winner, loser) values ({0},{1})".format(winner, loser))
    conn.commit()

    conn.close()</vul>
 
 
def swissPairings():
    """Returns a list of pairs of players for the next round of a match.
  
    Assuming that there are an even number of players registered, each player
    appears exactly once in the pairings.  Each player is paired with another
    player with an equal or nearly-equal win record, that is, a player adjacent
    to him or her in the standings.
  
    Returns:
      A list of tuples, each of which contains (id1, name1, id2, name2)
        id1: the first player's unique id
        name1: the first player's name
        id2: the second player's unique id
        name2: the second player's name
    """
    <vul/>conn = connect()
    c = conn.cursor()
    c.execute("select * from ranking order by count_wins desc")
    players_list = c.fetchall()
    num_games = len(players_list)/2
    result = []

    for game in range(num_games):
        first_player_index = game*2
        second_player_index = first_player_index + 1
        first_player_tuple = players_list[first_player_index]
        second_player_tuple = players_list[second_player_index]
        result.append((first_player_tuple[0], first_player_tuple[1], second_player_tuple[0], second_player_tuple[1]))
    conn.close()
    return result</vul>


