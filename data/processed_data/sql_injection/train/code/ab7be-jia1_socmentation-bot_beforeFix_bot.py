# Sources:
# Building a Chatbot using Telegram and Python (Part 1) by Gareth Dwyer

import json, logging, requests, time, urllib
from bisect import bisect
from db_helper import DBHelper

# See https://docs.python.org/3/library/logging.html#logging.basicConfig for basicConfig options and
# https://docs.python.org/3/library/logging.html#logrecord-attributes for format options
logging.basicConfig(filename = 'bot.log', format = "%(asctime)s %(levelname)s %(message)s", level = logging.INFO)

db = DBHelper()

with open('token.txt', 'r') as f:
    bot_token = f.readline().strip()

base_url = 'https://api.telegram.org/bot{}'.format(bot_token)

replies = {}
with open('replies.txt', 'r') as m:
    num_lines, command = m.readline().strip().split(' ')
    num_lines = int(num_lines)
    while num_lines:
        replies[command] = []
        for i in range(num_lines):
            replies[command].append(m.readline().strip())
        num_lines, command = m.readline().strip().split(' ')
        num_lines = int(num_lines)
logging.info("Reply messages loaded into memory")

blacklisted = {}
with open('blacklisted.txt', 'r') as f:
    num_blacklisted = int(f.readline().strip())
    for n in range(num_blacklisted):
        offender = int(f.readline().strip())
        if offender in blacklisted:
            blacklisted[offender] += 1
        else:
            blacklisted[offender] = 1
logging.info("Blacklisted senders loaded into memory")

reporting = {}
reporters_dict = {}
reporters_list = []
last_submitted_times = []
logging.info("Data structures loaded into memory")

# TODO: Remove idle reporters
timeout_oth = 300
timeout_ask = 180
logging.info("Response timeouts loaded into memory")

max_ans_len = 70
num_questions = len(replies['questions'])
<vul/>report_cooldown = 1800</vul>
logging.info("Other variables loaded into memory")

def get_json_from_url(url):
    response = requests.get(url)
    decoded_content = response.content.decode('utf-8')
    logging.info("GET %s responded with %s", url, decoded_content)
    return json.loads(decoded_content)

def get_updates(timeout, offset = None):
    url = '{}/getUpdates?timeout={}'.format(base_url, timeout)
    if offset:
        url += '&offset={}'.format(offset)
    return get_json_from_url(url)

def get_latest_update_id(updates):
    update_ids = []
    for update in updates['result']:
        update_ids.append(int(update['update_id']))
    latest_update_id = max(update_ids)
    logging.info("get_latest_update_id: Latest update ID is %d of %s", latest_update_id, update_ids)
    return latest_update_id

def get_latest_chat_id_and_text(updates):
    text = updates['result'][-1]['message']['text'].encode('utf-8')
    chat_id = updates['result'][-1]['message']['chat']['id']
    logging.info("get_latest_chat_id_and_text: Latest message is %s from chat %d", text, chat_id)
    return (text, chat_id)

def send_message(text, chat_id):
    text = urllib.parse.quote_plus(text)
    url = '{}/sendMessage?text={}&chat_id={}&parse_mode=Markdown'.format(base_url, text, chat_id)
    logging.info("send_message: Sending %s to chat %d", text, chat_id)
    requests.get(url)

def handle_updates(updates, latest_update_id):
    for update in updates['result']:
        try:
            text = update['message']['text']
            chat = update['message']['chat']['id']
            sender = update['message']['from']['id']
            is_ascii = all(ord(char) < 128 for char in text)
            logging.info("handle_updates: Received %s from %d", text.encode('utf-8'), sender)

            if not is_ascii:
                send_message(replies['invalid'][0], chat)
                continue

            if sender in reporting:
                if validate_answer(text):
                    reporting[sender].append(text)
                    reporting[sender][0] += 1
                    if reporting[sender][0] >= num_questions:
                        answers = reporting[sender][1:]
                        inserted, violations = db.insert(answers)
                        reporting.pop(sender)
                        if inserted:
                            <vul/>logging.info("handle_updates: Insert %s returns %r", str(answers), inserted)</vul>
                            send_message(replies['thanks'][0], chat)
                            last_submitted = int(time.time())
                            reporters_dict[sender] = last_submitted
                            reporters_list.append(sender)
                            last_submitted_times.append(last_submitted)
                        else:
                            send_message(replies['invalid'][0], chat)
                    else:
                        send_message(replies['questions'][reporting[sender][0]], chat)
                else:
                    send_message(replies['invalid'][0], chat)
                    send_message(replies['questions'][reporting[sender][0]], chat)
            elif text == '/help':
                send_message(replies[text][0], chat)
            elif text == '/start':
                send_message('\n'.join(replies[text]), chat)
            elif text == '/report':
                if sender in blacklisted:
                    send_message(replies['blacklisted'][0], chat)
                elif is_recent_reporter(sender):
                    <vul/>logging.info("handle_updates: %d not in blacklist", sender)</vul>
                    send_message(replies['cooldown'][0], chat)
                else:
                    send_message(replies[text][0], chat)
                    reporting[sender] = [0]
                    send_message(replies['questions'][0], chat)
            elif text == '/view':
                send_message(replies[text][0] + db.select_recent_pretty(), chat)
            else:
                send_message(replies['dk'][0], chat)
        except KeyError:
            pass

def is_recent_reporter(sender_id):
    global reporters_dict, reporters_list, last_submitted_times
    least_recent_index = bisect(last_submitted_times, int(time.time()) - report_cooldown)
    <vul/>for expired_reporter in range(least_recent_index):
        reporters_dict.pop(expired_reporter)</vul>
    last_submitted_times = last_submitted_times[least_recent_index:]
    reporters_list = reporters_list[least_recent_index:]
    is_recent = sender_id in reporters_dict
    logging.info("is_recent_reporter: %d returns %r", sender_id, is_recent)
    return is_recent

def validate_answer(ans):
    <vul/>too_long = len(ans) > max_ans_len
    logging.info("validate_answer: %s returns %r", str(ans), not too_long)
    return not too_long</vul>

def main():
    db.create_table()
    latest_update_id = None
    while True:
        updates = get_updates(timeout_oth, latest_update_id)
        if updates['result']:
            latest_update_id = get_latest_update_id(updates) + 1
            handle_updates(updates, latest_update_id)
        time.sleep(1)

if __name__ == '__main__':
    main()
