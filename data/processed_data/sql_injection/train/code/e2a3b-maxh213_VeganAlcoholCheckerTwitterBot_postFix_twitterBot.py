import tweepy
import secretConstants
import cgi
import time
from getAlcohol import getAlcoholByName
from lastReplied import getLastReplied, setLastReplied

auth = tweepy.OAuthHandler(secretConstants.CONSUMER_KEY, secretConstants.CONSUMER_SECRET)
auth.set_access_token(secretConstants.ACCESS_TOKEN, secretConstants.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

MESSAGE_BUFFER_SECONDS = 2.5

#alcoholName = "GUINESS"
#tweetAboutAlcohol(alcoholName)
#functions need to be declared above calling them it seems...

def formatReply(result):
    print (result)
    if result[2] == '':
        reply = result[0] + " is " + result[1] + "."
    elif result[0][2] != '' and result[2]:
        reply = result[0] + " brewed in " + result[2] + " is " + result[1] + "." 
    return reply

def getDMs():
    lastRepliedDmId = getLastReplied('DM')
    return api.direct_messages(full_text=True, since_id=lastRepliedDmId)

def replyToUnansweredDMs(dms):
    dms.reverse() #this is so old messages are answered first
    for dm in dms:
        print(dm.sender_screen_name + " sent " + dm.text)
        results = getAlcoholByName(dm.text)
        if len(results) > 10:
            replyToDm = "Sorry but I know a lot of alcohol with that in the name, could you be more specific?"
            api.send_direct_message(screen_name=dm.sender_screen_name, text=replyToDm)
        elif results == []:
            replyToDm = "Unfortunately I cannot find the name of the alcohol you specified in my database, apologies."
            api.send_direct_message(screen_name=dm.sender_screen_name, text=replyToDm)
        else:
            for result in results:
                time.sleep(MESSAGE_BUFFER_SECONDS)
                replyToDm = formatReply(result)
                api.send_direct_message(screen_name=dm.sender_screen_name, text=replyToDm)
        setLastReplied('DM', dm.id_str)
    
def main():
    dms = getDMs()
    <fix/>#for dm in dms:
    #    print(dm.text)
    #    setLastReplied('DM', dm.id_str)</fix>
    replyToUnansweredDMs(dms)


main()


def tweetAboutAlcohol(alcoholName):
    results = getAlcoholByName(alcoholName)

    tweetQueue = []
    for result in results: 
        status = formatReply(result)
        tweetQueue.append(status)

    for tweet in tweetQueue:
        api.update_status(status=tweet)
        print("tweeted: '" + tweet + "'")

