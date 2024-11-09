#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
if 'lib' not in sys.path:
    sys.path.append('lib')
import os, signal, logging, threading, re, traceback, time
import random
import zmq
from queue import Queue
import sup
import wzworkers as workers
from dataloader import DataLoader
from uniwipe import UniWipe
from wipeskel import *
import wzrpc
from beon import regexp
import pickle

from logging import config
from logconfig import logging_config
config.dictConfig(logging_config)
logger = logging.getLogger()

ctx = zmq.Context()
sig_addr = 'ipc://signals'
sig_sock = ctx.socket(zmq.PUB)
sig_sock.bind(sig_addr)

# Settings for you
domains = set() # d.witch_domains
targets = dict() # d.witch_targets
protected = set() # will be removed later
forums = dict() # target forums

# from lib import textgen
# with open('data.txt', 'rt') as f:
#     model = textgen.train(f.read())
# def mesasge():
#     while True:
#         s = textgen.generate_sentence(model)
#         try:
#             s.encode('cp1251')
#             break
#         except Exception:
#             continue
#     return s

def message():
    msg = []
    # msg.append('[video-youtube-'+
    #            random.choice(('3odl-KoNZwk', 'bu55q_3YtOY', '4YPiCeLwh5o',
    #                           'eSBybJGZoCU', 'ZtWTUt2RZh0', 'VXa9tXcMhXQ',))
    #            +']')
    msg.append('[image-original-none-http://simg4.gelbooru.com/'
               + '/images/db/1d/db1dfb62a40f5ced2043bb8966da9a98.png]')
    msg.append('Каждый хочет дружить с ядерной бомбой.')
    # msg.append('[video-youtube-'+random.choice(
    #     # ('WdDb_RId-xU', 'EFL1-fL-WtM', 'uAOoiIkFQq4',
    #     #  'eZO3K_4yceU', '1c1lT_HgJNo', 'WOkvVVaJ2Ks',
    #     #  'KYq90TEdxIE', 'rWBM2whL0bI', '0PDy_MKYo4A'))
    #     #('GabBLLOT6vw', 'qgvOpSquCAY', 'zUe-z9DZBNo', '4fCbfDEKZss', 'uIE-JgmkmdM'))
    #     ('42JQYPioVo4', 'jD6j072Ep1M', 'mPyF5ovoIVs', 'cEEi1BHycb0', 'PuA1Wf8nkxw',
    #      'ASJ9qlsPgHU', 'DP1ZDW9_xOo', 'bgSqH9LT-mI', ))
    # +']')
    # http://simg2.gelbooru.com//images/626/58ca1c9a8ffcdedd0e2eb6f33c9389cb7588f0d1.jpg
    # msg.append('Enjoy the view!')
    msg.append(str(random.randint(0, 9999999999)))
    return '\n'.join(msg)

def sbjfun():
    # return 'Out of the darkness we will rise, into the light we will dwell'
    return sup.randstr(1, 30)

# End
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--only-cache', '-C', action='store_true',
    help="Disables any requests in DataLoader (includes Witch)")
parser.add_argument('--no-shell', '-N', action='store_true',
    help="Sleep instead of starting the shell")
parser.add_argument('--tcount', '-t', type=int, default=10,
    help='WipeThread count')
parser.add_argument('--ecount', '-e', type=int, default=0,
    help='EvaluatorProxy count')
parser.add_argument('--upload-avatar', action='store_true', default=False,
    help='Upload random avatar after registration')
parser.add_argument('--av-dir', default='randav', help='Directory with avatars')
parser.add_argument('--rp-timeout', '-T', type=int, default=10,
    help='Default rp timeout in seconds')
parser.add_argument('--conlimit', type=int, default=3,
    help='http_request conlimit')
parser.add_argument('--noproxy-timeout', type=int, default=5,
    help='noproxy_rp timeout')

parser.add_argument('--caprate_minp', type=int, default=5,
    help='Cap rate minimum possible count for limit check')
parser.add_argument('--caprate_limit', type=float, default=0.8,
    help='Captcha rate limit')

parser.add_argument('--comment_successtimeout', type=float, default=0.8,
    help='Comment success timeout')
parser.add_argument('--topic_successtimeout', type=float, default=0.1,
    help='Topic success timeout')
parser.add_argument('--errortimeout', type=float, default=3,
    help='Error timeout')


parser.add_argument('--stop-on-closed', action='store_true', default=False,
    help='Forget about closed topics')
parser.add_argument('--die-on-neterror', action='store_true', default=False,
    help='Terminate spawn in case of too many NetErrors')

c = parser.parse_args()

# rps = {<vul/>}</vul>

noproxy_rp = sup.net.RequestPerformer()
noproxy_rp.proxy = ''
noproxy_rp.timeout = c.noproxy_timeout
noproxy_rp.timeout = c.rp_timeout

# rps[''] = noproxy_rp

# Achtung: DataLoader probably isn't thread-safe.
d = DataLoader(noproxy_rp, c.only_cache)
c.router_addr = d.addrs['rpcrouter']
noproxy_rp.useragent = random.choice(d.ua_list)

def terminate():
    logger.info('Shutdown initiated')
    # send_passthrough([b'GLOBAL', b'WZWorker', b'terminate'])
    send_to_wm([b'GLOBAL', b'WZWorker', b'terminate'])
    for t in threading.enumerate():
        if isinstance(t, threading.Timer):
            t.cancel()
    # try:
    #     wm.term()
    #     wm.join()
    # except: # WM instance is not created yet.
    #     pass
    logger.info('Exiting')

def interrupt_handler(signal, frame):
    pass # Just do nothing

def terminate_handler(signal, frame):
    terminate()

signal.signal(signal.SIGINT, interrupt_handler)
signal.signal(signal.SIGTERM, terminate_handler)

def make_net(proxy, proxytype):
    # if proxy in rps:
    #     return rps[proxy]
    net = sup.net.RequestPerformer()
    net.proxy = proxy
    if proxytype == 'HTTP' or proxytype == 'HTTPS':
        net.proxy_type = sup.proxytype.http
    elif proxytype == 'SOCKS4':
        net.proxy_type = sup.proxytype.socks4
    elif proxytype == 'SOCKS5':
        net.proxy_type = sup.proxytype.socks5
    else:
        raise TypeError('Invalid proxytype %s' % proxytype)
    # rps[proxy] = net
    net.useragent = random.choice(d.ua_list)
    net.timeout = c.rp_timeout
    return net

# UniWipe patching start
def upload_avatar(self, ud):
    if ('avatar_uploaded' in ud[0] and
        ud[0]['avatar_uploaded'] is True):
        return
    files = []
    for sd in os.walk(c.av_dir):
        files.extend(sd[2])
    av = os.path.join(sd[0], random.choice(files))
    self.log.info('Uploading %s as new avatar', av)
    self.site.uploadavatar('0', av)
    ud[0]['avatar'] = av
    ud[0]['avatar_uploaded'] = True

from lib.mailinator import Mailinator
# from lib.tempmail import TempMail as Mailinator

# Move this to WipeManager
def create_spawn(proxy, proxytype, pc, uq=None):
    for domain in domains:
        if domain in targets:
            tlist = targets[domain]
        else:
            tlist = list()
            targets[domain] = tlist
        if domain in forums:
            fset = forums[domain]
        else:
            fset = set()
            forums[domain] = fset
        net = make_net(proxy, proxytype)
        net.cookiefname = (proxy if proxy else 'noproxy')+'_'+domain
        w = UniWipe(fset, tlist, sbjfun, message, pc, net, domain, Mailinator,
            uq(domain) if uq else None)
        w.stoponclose = c.stop_on_closed
        w.die_on_neterror = c.die_on_neterror
        w.caprate_minp = c.caprate_minp
        w.caprate_limit = c.caprate_limit
        w.conlimit = c.conlimit
        w.comment_successtimeout = 0.2
        if c.upload_avatar:
            w.hooks['post_login'].append(upload_avatar)
        yield w

# UniWipe patching end

class WipeManager:
    def __init__(self, config, *args, **kvargs):
        super().__init__(*args, **kvargs)
        self.newproxyfile = 'newproxies.txt'
        self.proxylist = set()
        self.c = config
        self.threads = []
        self.processes = []
        self.th_sa = 'inproc://wm-wth.sock'
        self.th_ba = 'inproc://wm-back.sock'
        self.pr_sa = 'ipc://wm-wpr.sock'
        self.pr_ba = 'ipc://wm-back.sock'
        self.userqueues = {}
        self.usersfile = 'wm_users.pickle'
        self.targetsfile = 'wm_targets.pickle'
        self.bumplimitfile = 'wm_bumplimit.pickle'

    def init_th_sock(self):
        self.log.info(
            'Initializing intraprocess signal socket %s', self.th_sa)
        self.th_sock = self.p.ctx.socket(zmq.PUB)
        self.th_sock.bind(self.th_sa)

    def init_th_back_sock(self):
        self.log.info(
            'Initializing intraprocess backward socket %s', self.th_ba)
        self.th_back_sock = self.p.ctx.socket(zmq.ROUTER)
        self.th_back_sock.bind(self.th_ba)

    def init_pr_sock(self):
        self.log.info(
            'Initializing interprocess signal socket %s', self.pr_sa)
        self.pr_sock = self.p.ctx.socket(zmq.PUB)
        self.pr_sock.bind(self.pr_sa)

    def init_pr_back_sock(self):
        self.log.info(
            'Initializing interprocess backward socket %s', self.pr_ba)
        self.pr_back_sock = self.p.ctx.socket(zmq.ROUTER)
        self.pr_back_sock.bind(self.pr_ba)

    def read_newproxies(self):
        if not os.path.isfile(self.newproxyfile):
            return
        newproxies = set()
        with open(self.newproxyfile, 'rt') as f:
            for line in f:
                try:
                    line = line.rstrip('\n')
                    proxypair = tuple(line.split(' '))
                    if len(proxypair) < 2:
                        self.log.warning('Line %s has too few spaces', line)
                        continue
                    if len(proxypair) > 2:
                        self.log.debug('Line %s has too much spaces', line)
                        proxypair = (proxypair[0], proxypair[1])
                    newproxies.add(proxypair)
                except Exception as e:
                    self.log.exception('Line %s raised exception %s', line, e)
        # os.unlink(self.newproxyfile)
        return newproxies.difference(self.proxylist)

    def add_spawns(self, proxypairs):
        while self.running.is_set():
            try:
                try:
                    proxypair = proxypairs.pop()
                except Exception:
                    return
                self.proxylist.add(proxypair)
                for spawn in create_spawn(proxypair[0], proxypair[1], self.pc,
                        self.get_userqueue):
                    self.log.info('Created spawn %s', spawn.name)
                    self.spawnqueue.put(spawn, False)
            except Exception as e:
                self.log.exception('Exception "%s" raised on create_spawn', e)

    def spawn_workers(self, wclass, count, args=(), kvargs={}):
        wname = str(wclass.__name__)
        self.log.info('Starting %s(s)', wname)
        if issubclass(wclass, workers.WZWorkerThread):
            type_ = 0
            if not hasattr(self, 'th_sock'):
                self.init_th_sock()
            if not hasattr(self, 'th_back_sock'):
                self.init_th_back_sock()
        elif issubclass(wclass, workers.WZWorkerProcess):
            type_ = 1
            if not hasattr(self, 'pr_sock'):
                self.init_pr_sock()
            if not hasattr(self, 'pr_back_sock'):
                self.init_pr_back_sock()
        else:
            raise Exception('Unknown wclass type')
        for i in range(count):
            if not self.running.is_set():
                break
            try:
                w = wclass(*args, name='.'.join(
                    (wname, ('pr{0}' if type_ else 'th{0}').format(i))),
                    **kvargs)
                if type_ == 0:
                    self.threads.append(w)
                    w.start(self.p.ctx, self.th_sa)
                elif type_ == 1:
                    self.processes.append(w)
                    w.start(self.pr_sa)
            except Exception as e:
                self.log.exception('Exception "%s" raised on %s spawn',
                                   e, wname)

    def spawn_nworkers(self, type_, fun, count, args=(), kvargs={}):
        wname = str(fun.__name__)
        self.log.info('Starting %s(s)', wname)
        if type_ == 0:
            if not hasattr(self, 'th_sock'):
                self.init_th_sock()
            if not hasattr(self, 'th_back_sock'):
                self.init_th_back_sock()
        elif type_ == 1:
            if not hasattr(self, 'pr_sock'):
                self.init_pr_sock()
            if not hasattr(self, 'pr_back_sock'):
                self.init_pr_back_sock()
        else:
            raise Exception('Unknown wclass type')
        for i in range(count):
            if not self.running.is_set():
                break
            try:
                if type_ == 0:
                    w = workers.WZWorkerThread(
                        self.c.router_addr, fun, args, kvargs,
                        name='.'.join((wname, 'th{0}'.format(i))))
                    self.threads.append(w)
                    w.start(self.p.ctx, self.th_sa)
                elif type_ == 1:
                    w = workers.WZWorkerProcess(self.c.router_addr, fun, args, kvargs,
                        name='.'.join((wname, 'pr{0}'.format(i))))
                    self.processes.append(w)
                    w.start(self.pr_sa)
            except Exception as e:
                self.log.exception('Exception "%s" raised on %s spawn',
                                   e, wname)

    def spawn_wipethreads(self):
        return self.spawn_nworkers(0, WipeThread, self.c.tcount,
                                  (self.pc, self.spawnqueue))

    def spawn_evaluators(self):
        self.log.info('Initializing Evaluator')
        from evproxy import EvaluatorProxy
        def ev_init():
            from lib.evaluators.PyQt4Evaluator import Evaluator
            return Evaluator()
        return self.spawn_nworkers(1, EvaluatorProxy, self.c.ecount,
                                  (ev_init,))

    def load_users(self):
        if not os.path.isfile(self.usersfile):
            return
        with open(self.usersfile, 'rb') as f:
            users = pickle.loads(f.read())
        try:
            for domain in users.keys():
                uq = Queue()
                for ud in users[domain]:
                    self.log.debug('Loaded user %s:%s', domain, ud['login'])
                    uq.put(ud)
                self.userqueues[domain] = uq
        except Exception as e:
            self.log.exception(e)
            self.log.error('Failed to load users')

    def save_users(self):
        users = {}
        for d, uq in self.userqueues.items():
            uqsize = uq.qsize()
            uds = []
            for i in range(uqsize):
                uds.append(uq.get(False))
            users[d] = uds
        with open(self.usersfile, 'wb') as f:
            f.write(pickle.dumps(users, pickle.HIGHEST_PROTOCOL))
        self.log.info('Saved users')

    def get_userqueue(self, domain):
        try:
            uq = self.userqueues[domain]
        except KeyError:
            self.log.info('Created userqueue for %s', domain)
            uq = Queue()
            self.userqueues[domain] = uq
        return uq

    def load_targets(self):
        fname = self.targetsfile
        if not os.path.isfile(fname):
            return
        with open(fname, 'rb') as f:
            data = pickle.loads(f.read())
        if 'targets' in data:
            self.log.debug('Target list was loaded')
            targets.update(data['targets'])
        if 'forums' in data:
            self.log.debug('Forum set was loaded')
            forums.update(data['forums'])
        if 'domains' in data:
            self.log.debug('Domain set was loaded')
            domains.update(data['domains'])
        if 'sets' in data:
            self.log.debug('Other sets were loaded')
            self.pc.sets.update(data['sets'])

    def load_bumplimit_set(self):
        if not os.path.isfile(self.bumplimitfile):
            return
        with open(self.bumplimitfile, 'rb') as f:
            self.pc.sets['bumplimit'].update(pickle.loads(f.read()))

    def save_targets(self):
        data = {
            'targets': targets,
            'forums': forums,
            'domains': domains,
            'sets': self.pc.sets,
            }
        with open(self.targetsfile, 'wb') as f:
            f.write(pickle.dumps(data, pickle.HIGHEST_PROTOCOL))

    def targets_from_witch(self):
        for t in d.witch_targets:
            if t['domain'] == 'beon.ru' and t['forum'] == 'anonymous':
                try:
                    add_target_exc(t['id'], t['user'])
                except ValueError:
                    pass

    def terminate(self):
        msg = [b'GLOBAL']
        msg.extend(wzrpc.make_sig_msg(b'WZWorker', b'terminate', []))
        if hasattr(self, 'th_sock'):
            self.th_sock.send_multipart(msg)
        if hasattr(self, 'pr_sock'):
            self.pr_sock.send_multipart(msg)

    def join_threads(self):
        for t in self.threads:
            t.join()

    def send_passthrough(self, interface, method, frames):
        msg = [frames[0]]
        msg.extend(wzrpc.make_sig_msg(frames[1], frames[2], frames[3:]))
        self.th_sock.send_multipart(msg)
        self.pr_sock.send_multipart(msg)

    def __call__(self, parent):
        self.p = parent
        self.log = parent.log
        self.inter_sleep = parent.inter_sleep
        self.running = parent.running
        self.p.sig_sock.setsockopt(zmq.SUBSCRIBE, b'WipeManager')
        self.p.wz.set_sig_handler(b'WipeManager', b'passthrough', self.send_passthrough)
        if self.c.tcount > 0:
            self.pc = ProcessContext(self.p.name, self.p.ctx,
                self.c.router_addr, noproxy_rp)
            self.spawnqueue = Queue()
            self.load_bumplimit_set()
            self.load_targets()
            self.load_users()
            self.spawn_wipethreads()
        if self.c.ecount > 0:
            self.spawn_evaluators()
        try:
            while self.running.is_set():
                # self.targets_from_witch()
                if self.c.tcount == 0:
                    self.inter_sleep(5)
                    continue
                self.pc.check_waiting()
                new = self.read_newproxies()
                if not new:
                    self.inter_sleep(5)
                    continue
                self.add_spawns(new)
        except WorkerInterrupt:
            pass
        except Exception as e:
            self.log.exception(e)
        self.terminate()
        self.join_threads()
        if self.c.tcount > 0:
            self.save_users()
            self.save_targets()

wm = workers.WZWorkerThread(c.router_addr, WipeManager, (c,),
    name='SpaghettiMonster')
wm.start(ctx, sig_addr)

def add_target(domain, id_, tuser=None):
    if domain not in targets:
        targets[domain] = []
    tlist = targets[domain]
    id_ = str(id_)
    tuser = tuser or ''
    t = (tuser, id_)
    logger.info('Appending %s to targets[%s]', repr(t), domain)
    tlist.append(t)

def remove_target(domain, id_, tuser=None):
    tlist = targets[domain]
    id_ = str(id_)
    tuser = tuser or ''
    t = (tuser, id_)
    logger.info('Removing %s from targets[%s]', repr(t), domain)
    tlist.remove(t)

def add_target_exc(domain, id_, tuser=None):
    if domain not in targets:
        targets[domain] = []
    tlist = targets[domain]
    id_ = str(id_)
    tuser = tuser or ''
    t = (tuser, id_)
    if t in protected:
        raise ValueError('%s is protected' % repr(t))
    if t not in tlist:
        logger.info('Appending %s to targets[%s]', repr(t), domain)
        tlist.append(t)

r_di = re.compile(regexp.f_udi)

def atfu(urls):
    for user, domain, id1, id2 in r_di.findall(urls):
        id_ = id1+id2
        add_target(domain, id_, user)

def rtfu(urls):
    for user, domain, id1, id2 in r_di.findall(urls):
        id_ = id1+id2
        remove_target(domain, id_, user)

def get_forum_id(name):
    id_ = d.bm_id_forum.get_key(name)
    int(id_, 10)  # id is int with base 10
    return id_

# def aftw(name):
#     id_ = get_forum_id(name)
#     logger.info('Appending %s (%s) to forums', name, id_)
#     forums.append(id_)

# def rffw(name):
#     id_ = get_forum_id(name)
#     logger.info('Removing %s (%s) from forums', name, id_)
#     forums.remove(id_)

# def aftw(name):
#     id_ = get_forum_id(name)
#     logger.info('Appending %s to forums', name)
#     forums.add(name)

# def rffw(name):
#     id_ = get_forum_id(name)
#     logger.info('Removing %s from forums', name)
#     forums.remove(name)

r_udf = re.compile(regexp.udf_prefix)

def affu(urls):
    for user, domain, forum in r_udf.findall(urls):
        if domain not in forums:
            forums[domain] = set()
        if len(forum) > 0:
            get_forum_id(forum)
        logger.info('Appending %s:%s to forums[%s]', user, forum, domain)
        forums[domain].add((user, forum))

def rffu(urls):
    for user, domain, forum in r_udf.findall(urls):
        if len(forum) > 0:
            get_forum_id(forum)
        logger.info('Removing %s:%s from forums[%s]', user, forum, domain)
        forums[domain].remove((user, forum))

def add_user(domain, login, passwd):
    uq = wm.get_userqueue(domain)
    uq.put({'login': login, 'passwd': passwd}, False)

def send_to_wm(frames):
    msg = [frames[0]]
    msg.extend(wzrpc.make_sig_msg(frames[1], frames[2], frames[3:]))
    sig_sock.send_multipart(msg)

def send_passthrough(frames):
    msg = [b'WipeManager']
    msg.extend(wzrpc.make_sig_msg(b'WipeManager', b'passthrough', frames))
    sig_sock.send_multipart(msg)

def drop_users():
    send_passthrough([b'WipeSkel', b'WipeSkel', b'drop-user'])

def log_spawn_name():
    send_passthrough([b'WipeThread', b'WipeThread', b'log-spawn-name'])

<vul/>if c.no_shell:
    while True:
        time.sleep(1)
else:
    try:
        import IPython</vul>
        IPython.embed()
    <vul/>except ImportError:
        # fallback shell</vul>
        while True:
            try:
                exec(input('> '))
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
            except SystemExit:
                break
            except:
                print(traceback.format_exc())

terminate()
