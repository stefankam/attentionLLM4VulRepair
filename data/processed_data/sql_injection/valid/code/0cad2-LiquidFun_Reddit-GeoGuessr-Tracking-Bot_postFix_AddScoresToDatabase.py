import praw
import re
from datetime import datetime
import operator

import sqlite3

def getDate(submission):
    time = datetime.fromtimestamp(submission.created)
    # return datetime.date.fromtimestamp(time)
    return time.strftime('%Y-%m-%d %H:%M:%S')

def getTitle(submission):
    delimChars = ['-', ':', '=', '#', '(', ')']

    title = str(submission.title)

    # Get first part of title by spliting before line or colon
    for delimChar in delimChars:
        title = title.split(delimChar)[0]

    # Join the resulting chars
    return str(''.join(re.findall('[a-zA-Z]', title)).lower())

def addToDatabase(submissionList):

    # Measure time
    startTime = datetime.now()

    database = sqlite3.connect('database.db')
    cursor = database.cursor()

    #reddit = getRedditInstance()

    botUsername = getBotUsername()

    # Get top level comments from submissions and get their first numbers with regex
    for submission in reversed(list(submissionList)):
        scoresInChallenge = [[-1, ''], [-2, ''], [-3, ''], [-4, '']] 
        for topLevelComment in submission.comments:

            # Looks for !TrackThisSeries and !StopTracking posts and replies to them
            try:
                if topLevelComment.author.name == submission.author.name:
                    alreadyReplied = False
                    if '!trackthisseries' in topLevelComment.body.lower():
                        print("Found track request: " + str(submission.id))
                        # Write new entries to the local database
                        <fix/>cursor.execute("INSERT OR REPLACE INTO SeriesTracking VALUES (?, ?)", [getTitle(submission), getDate(submission)])</fix>
                        
                        for reply in topLevelComment.replies:
                            if reply.author.name == botUsername:
                                alreadyReplied = True
                                #cursor.execute("INSERT OR REPLACE INTO TrackingRequests VALUES ('" + str(topLevelComment.fullname) + "')")

                        #if cursor.execute("SELECT COUNT(*) FROM TrackingRequests WHERE CommentID = '" + str(topLevelComment.fullname) + "'").fetchone()[0] == 0:
                        if not alreadyReplied:
                            replyToTrackRequest(topLevelComment, True)
                    if '!stoptracking' in topLevelComment.body.lower():
                        print("Found stop tracking request: " + str(submission.id))
                        # Delete old entries in the database
                        <fix/>cursor.execute("DELETE FROM SeriesTracking WHERE SeriesTitle = ?", [getTitle(submission)])</fix>
                        
                        for reply in topLevelComment.replies:
                            if reply.author.name == botUsername:
                                alreadyReplied = True
                                #cursor.execute("INSERT OR REPLACE INTO TrackingRequests VALUES ('" + str(topLevelComment.fullname) + "')")

                        #if cursor.execute("SELECT COUNT(*) FROM TrackingRequests WHERE CommentID = '" + str(topLevelComment.fullname) + "'").fetchone()[0] == 0:
                        if not alreadyReplied:
                            replyToTrackRequest(topLevelComment, False)
            except AttributeError:
                pass

            # Avoid comments which do not post their own score; Get the highest number in each comment and add it to the list with the user's username
            if 'Previous win:' not in topLevelComment.body and 'for winning' not in topLevelComment.body and 'for tying' not in topLevelComment.body and '|' not in topLevelComment.body and topLevelComment is not None and topLevelComment.author is not None:
                try:
                    number = max([int(number.replace(',', '')) for number in re.findall('(?<!round )(?<!~~)(?<!\w)\d+\,?\d+', topLevelComment.body)])
                except (IndexError, ValueError) as e:
                    number = -1
                    break
                if 0 <= number <= 32395:
                    scoresInChallenge.append([int(number), topLevelComment.author.name])
        scoresInChallenge.sort(key = operator.itemgetter(0), reverse = True)

        # If two players have the same score add the second one to the authors of the first challenge with a pipe character inbetween
        for i in range(0, 3):
            while scoresInChallenge[i][0] == scoresInChallenge[i + 1][0]:
                scoresInChallenge[i][1] += "|" + scoresInChallenge[i + 1][1]
                del scoresInChallenge[i + 1]
        #print(index)
        #print(getTitle(submission.title))
        #print(submission.id)
        #print(scoresInChallenge[0][1])
        #print(scoresInChallenge[1][1])
        #print(scoresInChallenge[2][1])
        #print(submission.created)

        # Write new entries to the local database
        <fix/>if getTitle(submission) != '':
            record = (str(submission.id), getTitle(submission), str(scoresInChallenge[0][1]), str(scoresInChallenge[1][1]), str(scoresInChallenge[2][1]), getDate(submission))
            cursor.execute("INSERT OR REPLACE INTO ChallengeRankings VALUES (?, ?, ?, ?, ?, ?)", record)</fix>

        #if cursor.execute("SELECT COUNT(*) FROM ChallengeRankings WHERE SubmissionID = '" + submission.id + "'").fetchone()[0] == 0:
        #    cursor.execute("INSERT INTO ChallengeRankings VALUES (?, ?, ?, ?, ?, ?)", record)
        # Update existing entries in the local database
        #else:
        #    cursor.execute("UPDATE ChallengeRankings SET Place1 = '" + str(scoresInChallenge[0][1]) + "', Place2 = '" + str(scoresInChallenge[1][1]) + "', Place3 = '" + str(scoresInChallenge[2][1]) + "' WHERE SubmissionID = '" + str(submission.id) + "'")

    database.commit()
    database.close()

# Reply to the comment which asks the bot to track the series
def replyToTrackRequest(comment, positive):
    if positive == True:
        print("I will be tracking this series: " + getTitle(comment.submission.title) + " because of this comment " + comment.fullname)
        #comment.reply("I will be tracking this series from now on.")
    else:
        print("I will stop tracking this series: " + getTitle(comment.submission.title) + " because of this comment " + comment.fullname)
        #comment.reply("I will stop tracking this series from now on.")

def getBotUsername():
    inputFile = open("RedditAPIAccess.txt")
    lines = []
    for line in inputFile:
        lines.append(line)
    return line[2]