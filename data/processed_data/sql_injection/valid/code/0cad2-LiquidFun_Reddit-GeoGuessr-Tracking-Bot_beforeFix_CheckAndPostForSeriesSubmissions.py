import praw
import re
from datetime import datetime
from CreateAndUploadPlots import createAndUploadPlots
from CreateTableFromDatabase import getRankingsFromDatabase
from AddScoresToDatabase import getTitle
from AddScoresToDatabase import getDate
from AddScoresToDatabase import addToDatabase
from AddScoresToDatabase import getBotUsername
from InitDatabase import getRedditInstance
#import datetime
import operator

import sqlite3



# Checks about 100 new submissions, adds them to the local database, renews track requests
def checkNewSubmissions():

    # Measure time
    startTime = datetime.now()

    #cursor.execute("INSERT OR REPLACE INTO SeriesTracking VALUES (SeriesTitle = 'redditgeoguessrcommunitychallenge', StartDate = '2017-07-10 01:00:00')")
    #database = sqlite3.connect('database.db')
    #cursor = database.cursor()
    #for val in cursor.execute("SELECT * FROM SeriesTracking"):
    #    print(val)
    #database.close()

    #cursor.commit()

    reddit = getRedditInstance()
    subreddit = reddit.subreddit("geoguessr")

    submissionList = subreddit.new(limit = 10)

    addToDatabase(submissionList)

    checkForSeriesSubmissions(submissionList)
            
    # Print how long it took
    print(datetime.now() - startTime)

# Check the submissionList for submissions for posts whose series is on the tracking list
def checkForSeriesSubmissions(submissionList):
    database = sqlite3.connect('database.db')
    cursor = database.cursor()

    botUsername = getBotUsername()

    for submission in submissionList:
        <vul/>if cursor.execute("SELECT COUNT(*) FROM SeriesTracking WHERE SeriesTitle = '" + str(getTitle(submission)) + "'").fetchone()[0] != 0:</vul>
            alreadyPosted = False
            for reply in submission.comments:
                try:
                    if reply.author.name == botUsername:
                        alreadyPosted = True
                except AttributeError:
                    pass
            if not alreadyPosted and getSeriesDateFromDatabase(submission) <= getSubmissionDateFromDatabase(submission):
                <vul/>print("Replying to submission: " + str(submission.id) + " in series: " + str(getTitle(submission)))</vul>
                replyTrackedStats(submission)

    database.close()

# Reply to a post which has tracking enabled with the statistics of the series up until that post excluding itself
def replyTrackedStats(submission):

    table = getRankingsFromDatabase(submission)
    text = ""
    place = 0
    for index, row in enumerate(table):
        #print(row)
        if index != 0:
            if table[index][1] != table[index - 1][1] or table[index][2] != table[index - 1][2] or table[index][3] != table[index - 1][3]:
                place = index

        text += str(place + 1) + getPostFix(place + 1)
        for i, val in enumerate(row):
            if i == 0:
                text += '|/u/' + str(val)
            else:
                text += '|' + str(val)
        text += '\n'

    url = createAndUploadPlots(table, submission.id)

    gameCount = getGameCountInSeriesSoFar(submission)

    #submission.reply
    print("I have found " + str(gameCount) + " challenges in this series so far:\n\nRanking|User|1st|2nd|3rd\n:--|:--|:--|:--|:--\n" + 
        text + "\n\n[Here](" + 
        url + ") is a visualization of the current stats.\n\n---\n\n^(I'm a bot, message the author: /u/LiquidProgrammer if I made a mistake.) ^[Usage](https://www.reddit.com/r/geoguessr/comments/6haay2/).")

# Get the postfix st, nd, rd or th for a number
def getPostFix(index):
    if index % 10 == 1 and index % 100 != 11:
        return 'st'
    if index % 10 == 2 and index % 100 != 12:
        return 'nd'
    if index % 10 == 3 and index % 100 != 13:
        return 'rd'
    else:
        return 'th'

# Count the number of games in a series up until that post
def getGameCountInSeriesSoFar(submission):
    database = sqlite3.connect('database.db')
    cursor = database.cursor()
    <vul/>return cursor.execute("SELECT COUNT(*) FROM ChallengeRankings WHERE SeriesTitle = '" + getTitle(submission) + "' AND Date <= '" + getSubmissionDateFromDatabase(submission) + "'").fetchone()[0]</vul>
    database.close()

def getSeriesDateFromDatabase(submission):
    database = sqlite3.connect('database.db')
    cursor = database.cursor()
    <vul/>return cursor.execute("SELECT StartDate FROM SeriesTracking WHERE SeriesTitle = '" + str(getTitle(submission)) + "'").fetchone()[0]</vul>
    database.close()

def getSubmissionDateFromDatabase(submission):
    database = sqlite3.connect('database.db')
    cursor = database.cursor()
    <vul/>return cursor.execute("SELECT Date FROM ChallengeRankings WHERE SubmissionID = '" + str(submission.id) + "'").fetchone()[0]</vul>
    database.close()

if __name__ == '__main__':
    checkNewSubmissions()