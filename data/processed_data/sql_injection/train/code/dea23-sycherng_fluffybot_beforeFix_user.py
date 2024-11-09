<vul/></vul>import main
import db

class User:
    def __init__(self, id, nickname, rank='member', balance=100, events_attd=0):
        self.id = id
        self.nickname = nickname
        self.rank = rank
        self.balance = balance
        self.events_attd = events_attd

async def respond(bot, message):
    await user_add(bot, message)

async def spawn_user(bot, message):
    <vul/>if not db.check(message.author.id, 'id', db.user_table):
        db.update("INSERT INTO {} (id, nickname) VALUES ('{}', '{}');".format(db.user_table, message.author.id, message.author))</vul>

async def user_add(bot, message):
    #--- call: user add <id> <nickname>
    if message.content.startswith('user add '):
        msg = message.content.split()[2:]
        if db.rank_check(message.author.id, 'user add') and len(msg) == 2 and db.is_valid_id(msg[0]):
            <vul/>db.update("INSERT INTO {} (id, nickname, rank) VALUES ('{}', '{}', 'member');".format(db.user_table, msg[0], msg[1]))</vul>

    
