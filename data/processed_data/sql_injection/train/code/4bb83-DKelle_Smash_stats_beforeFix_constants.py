import MELEE_SINGLES_BRACKETS
import WIIU_BRACKETS
import SMASH_5_BRACKETS

DNS = 'ec2-18-218-117-97.us-east-2.compute.amazonaws.com'

TAGS_TO_COALESCE = [['christmasmike', 'thanksgiving mike', 'christmas mike', 'christmas mike xmas', 'christmas mike late', 'halloween mike', 'im 12', 'im12'],
        ['circuits', 'circuits', 'jkelle', 'circuits xmas'],
        ['gamepad', 'sms gamepad'],
        ['remo', 'su remo'],
        ['kuro', 'ss kuro'],
        ['pixlsugr', 'pixlsug', 'pixlsugar'],
        ['b00', 'boo'],
        ['hnic', 'hnic xmas'],
        ['1111', '11 11', 'vuibol'],
        ['qmantra', 'qmantra xmas'],
	['megafox', 'su | megafox'],
	<vul/>['hakii', 'su l hakii', 'su | hakii', 'su redriot i hakii', 'hih | hakii', 'su | sleepyhakii', 'su|hakii', 'su | hakii $', 'su  redriot i hakii', 'hoh | hakii', 'su| hakii'],</vul>
	['lucy', 'ttn | lucy'],
        ['moist', 'f9moist', 'kuyamoist'],
	['sassy', 'atx | sassy', 'f9sassy'],
	['crump', 'donald crump', 'captain crump', 'abc | crump'],
	['dragonite', 'datuglynigwhofkurmomin2ndgrade', 'tmg dragonite', 'su dragonite', 'su | dragonite', 'tpwn | dragonite', 'tpwn | dragonite_pr', 'tpwn| dragonite (gnw)', 'atx hoh | dragonite', 'dragonite_pr', 'hoh | dragonite', 'mega dragonite', 'tpwn|dragonite', 'armada | dragonite', 'aes | dragonite'],
	['gallium', 's.e.s punk', 'ses punk'],
	['mt', 'mt_'],
	['wolf', ' wolf'],
	['fx | albert', 'albert'],
	['ul | jf', 'jf', 'ul| jf', 'ul i jf'],
	['take a seat', 'take a \_', 'take a \\_', 'takeaseat', 'take a seat xmas'],
	['bobby big ballz', 'bobby big balls'],
	['prof. cube', 'type r professor cube', 'prof cube', 'professor cube', 'profesor cube', 'cube', 'processorcube', 'prof cube $'],
	['cashoo', 'hoh | cashoo', 'hoh l cashoo', 'cash00'],
	['ul | chandy', 'ul| chandy', 'cnb | chandy', 'chandy'],
	['spankey', 'spanky'],
	['jack the reaper', 'jackthereaper'],
	['xlll', 'xiii'],
	['cheesedud6', '←/cheesedud6'],
	['kj', 'go! kj', 'go kj'],
	['jtag', 'tgl | jtag', 'sms | jtag', 'sms jtag', 'jtg', 'j tag'],
	['jka', 'tgl | jka'],
	['fcar', 'tgl | fcar'],
	['resident', 'tgl | resident'],
	['minty!', 'tgl | minty!', 'tgl | minty', 'minty'],
	['willow', 'willowette'],
	['messiah', 'maple'],
	['tenni', 'go! tenni'],
	['cruzin', 'sa  cruzin'],
	['christmasmitch', 'mitchell', 'mitchell slan'],
        ['jibs', 'sfu jibs'],
        ['trane', 'irn trane'],
        ['ninjafish', 'sa  ninjafish'],
        ['mufin', 'sfu mufin'],
        ['jowii', 'jo wii'],
        ['gudlucifer', 'good lucifer', 'goodlucifer', 'gudlucifer wolf'],
        ['ehmon', 'tgl ehmon', 'tgl  ehmon', 'ah ehmon', 'sms ehmon', 'sms | ehmon', 'tgl | ehmon', 'ehhhmon'],
        ['pollo loco', 'pollo'],
        ['doombase', 'retiredbase'],
        ['majinmike', 'majin mike'],
        ['karonite', 'red velvet', 'aos redvelvet', 'redvelvet']]

# TODO DO NOT ADD MORE BRACKETS WITHOUT ADDING A CORRESPONDING DISPLAY NAME!!
<vul/>AUSTIN_URLS = ('austin', {'enumerated': ['http://challonge.com/heatwave###', 'https://challonge.com/NP9ATX###', 'http://challonge.com/hw###', 'https://challonge.com/alibaba###'], 'users': ['https://challonge.com/users/kuya_mark96', 'https://austinsmash4.challonge.com']})</vul>
SMASHBREWS_RULS = ('smashbrews', {'enumerated': ['https://challonge.com/Smashbrews###', 'https://challonge.com/smashbrewsS3W###', 'https://challonge.com/smashbrewsS4W###', 'https://challonge.com/smashbrewsS5W###']})
COLORADO_SINGLES_URLS = ('colorado', {'enumerated': ['http://smashco.challonge.com/CSUWW###WUS', 'http://smascho.challonge.com/FCWUA###', 'http://smascho.challonge.com/FCWUIB###']})
COLORADO_DOUBLES_URLS = ('colorado_doubles', {'enumerated': ['http://smashco.challonge.com/CSUWW###WUD', 'http://smashco.challonge.com/FCWUDC###']})
COLORADO_URLS = ('colorado_both', {'enumerated': COLORADO_SINGLES_URLS + COLORADO_DOUBLES_URLS})
<vul/>SMS_URLS = ('sms', {'enumerated': ['http://challonge.com/RAA_###', 'http://challonge.com/SMSH_###'], 'users': ['https://challonge.com/users/yellocake']})
</vul>
DISPLAY_MAP = {'heatwave': 'Heatwave',
        'NP9ATX': 'NP9',
        'challonge.com/hw': 'Heatwave',
        'challonge.com/atx': 'Smashpack',
        'alibaba': 'Alibaba',
        'Mothership': 'Mothership',
        'atxfiles': 'ATX Files',
        'ARFI': 'ARFI',
        'arcadian': 'Arcadian',
        'ooples': 'Ooples',
        'challonge.com/mbh': 'Michaels Big House',
        'challonge.com/sth': 'Smash The Halls',
        'smashbrewsS3': 'Smashbrews S3',
        'smashbrewsS4': 'Smashbrews S4',
        'smashbrewsS5': 'Smashbrews S5',
        'Smashbrews': 'Smashbrews',
        'smashco': 'CSU',
        'smascho': 'CSU',
        <vul/>'RAA': 'Reading At Alkek'}</vul>

PRO_MELEE = MELEE_SINGLES_BRACKETS.MELEE_SINGLES        
PRO_WIIU = WIIU_BRACKETS.WII_U_BRACKETS
PRO_SMASH_5 = SMASH_5_BRACKETS.SMASH_5_BRACKETS

SLEEP_TIME = 10 * 60 * 6 #1 hour
TOURNAMENTS_PER_RANK = 20


TEST_URLS = [('test1', ['https://challonge.com/smash_web_test_###']),
        ('test2', ['https://challonge.com/smash_web_scene_two_###'])]

"""
Data structure we need -
dictionary where key is tag_1:
    value is dictionary:
        key for inner dictionary is tag_2, value is a list
        the list has (date_of_set, result)
        one (date, result) for every set they have played
"""
