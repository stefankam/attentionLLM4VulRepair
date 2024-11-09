import asyncio
import main
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
    <fix/>if not db.check(message.author.id, 'id', db.users):
        db.update("INSERT INTO {} (id, nickname) VALUES (%s, %s);".format(db.users), (message.author.id, message.author.name))</fix>

async def user_add(bot, message):
    #--- call: user add <id> <nickname>
    if message.content.startswith('user add '):
        msg = message.content.split()[2:]
        if db.rank_check(message.author.id, 'user add') and len(msg) == 2 and db.is_valid_id(msg[0]):
            <fix/>db.update("INSERT INTO {} (id, nickname, rank) VALUES (%s, %s, 'member');".format(db.users), (msg[0], msg[1]))</fix>

<fix/>async def user_edit(bot, message):
    #--- call: user edit <id> <attribute> <value>
    if message.content.startswith('user edit '):
        msg = message.content.split()[2:]
        if db.rank_check(message.author.id, 'user edit') and len(msg) == 3 and db.is_valid_id(msg[0]):
            pass</fix>
