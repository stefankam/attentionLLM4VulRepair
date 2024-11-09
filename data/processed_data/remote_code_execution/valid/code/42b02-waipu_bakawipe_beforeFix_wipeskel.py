# -*- coding: utf-8 -*-
# -*- mode: python -*-
import logging, re
from queue import Queue, Empty
import zmq
import beon, sup, wzrpc
from beon import regexp
from wzworkers import WorkerInterrupt
from ocr import OCRError, PermOCRError, TempOCRError
from sup.ticker import Ticker
from userdata import short_wordsgen
from enum import Enum
from collections import Counter, deque

class ProcessContext:
    def __init__(self, name, ctx, wz_addr, noproxy_rp):
        self.log = logging.getLogger('.'.join((name, type(self).__name__)))
        self.zmq_ctx = ctx
        self.ticker = Ticker()
        self.sets = {}
        self.sets['targets'] = set()
        self.sets['waiting'] = dict()
        self.sets['pending'] = set()
        self.sets['closed'] = set()
        self.sets['bumplimit'] = set()
        self.sets['protected'] = set()
        self.wz_addr = wz_addr
        self.noproxy_rp = noproxy_rp

    def make_wz_sock(self):
        self.log.debug('Initializing WZRPC socket')
        wz_sock = self.zmq_ctx.socket(zmq.DEALER)
        wz_sock.setsockopt(zmq.IPV6, True)
        wz_sock.connect(self.wz_addr)
        return wz_sock

    def check_waiting(self):
        elapsed = self.ticker.elapsed()
        waiting = self.sets['waiting']
        for k, v in waiting.copy().items():
            rem = v - elapsed
            if rem <= 0:
                del waiting[k]
                self.log.info('Removing %s from %s', k[0], k[1])
                try:
                    self.sets[k[1]].remove(k[0])
                except KeyError:
                    self.log.error('No %s in %s', k[0], k[1])
            else:
                waiting[k] = rem

    def add_waiting(self, sname, item, ttl):
        self.sets['waiting'][(item, sname)] = ttl

class WTState(Enum):
    null = 0
    starting = 2
    empty = 3
    sleeping = 4
    running = 5

class WipeState(Enum):
    null = 0
    starting = 2
    terminating = 3
    sleeping = 4
    running = 5

    logging_in = 6
    post_login_hooks = 7
    registering = 8
    pre_register_hooks = 9
    post_register_hooks = 10
    deobfuscating_capage = 11
    solving_captcha = 12
    reporting_code = 13

    operation = 50
    waiting_for_targets = 51
    scanning_for_targets = 52
    posting_comment = 53
    posting_topic = 54

class state:
    def __init__(self, defstate):
        self.defstate = defstate
        self.state = defstate

    def __call__(self, state):
        self.state = state
        return self

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        self.state = self.defstate

    @property
    def name(self):
        return self.state.name

    @property
    def value(self):
        return self.state.value

class cstate:
    def __init__(self, obj, state):
        self.obj = obj
        self.backstate = obj.state
        self.newstate = state

    def __enter__(self):
        self.obj.log.info('Switching state to %s', repr(self.newstate))
        self.obj.state = self.newstate

    def __exit__(self, exception_type, exception_value, traceback):
        self.obj.log.info('Switching state to %s', repr(self.backstate))
        self.obj.state = self.backstate


class WipeThread:
    def __init__(self, pc, spawnqueue, *args, **kvargs):
        self.pc = pc
        self.spawnqueue = spawnqueue
        self.spawn = None
        self.state = WTState.null
        self.wz_reply = None

    def deobfuscate_capage(self, domain, page):
        result = []
        def accept(that, reqid, seqnum, status, data):
            if status == wzrpc.status.success or status == wzrpc.status.error:
                <vul/>result.extend(map(lambda x:x.decode('utf-8'), data))</vul>
            elif status == wzrpc.status.e_req_denied:
                <vul/>self.log.warn('Status {0}, reauthentificating'.\</vul>
                    format(wzrpc.name_status(status)))
                self.p.auth_requests()
                that.retry = True
            elif status == wzrpc.status.e_timeout:
                self.log.warn('Timeout {0}, retrying'.format(data[0]))
                that.retry = True
            else:
                self.log.warn('Status {0}, retrying'.format(wzrpc.name_status(status)))
                that.retry = True
        self.p.wz_wait_reply(accept,
            b'Evaluator', b'evaluate', (domain.encode('utf-8'), page.encode('utf-8')),
            timeout=60)
        return tuple(result)

    def solve_captcha(self, img):
        result = []
        def accept(that, reqid, seqnum, status, data):
            if status == wzrpc.status.success or status == wzrpc.status.error:
                result.extend(map(lambda x:x.decode('utf-8'), data))
            elif status == wzrpc.status.e_req_denied:
                self.log.warn('Status {0}, reauthentificating'.\
                    format(wzrpc.name_status(status)))
                self.p.auth_requests()
                that.retry = True
            elif status == wzrpc.status.e_timeout:
                self.log.warn('Timeout {0}, retrying'.format(data[0]))
                that.retry = True
            else:
                self.log.warn('Status {0}, retrying'.format(wzrpc.name_status(status)))
                that.retry = True
        self.p.wz_wait_reply(accept,
            b'Solver', b'solve', (b'inbound', img), timeout=300)
        if len(result) == 2: # Lame and redundant check. Rewrite this part someday.
            return result
        else:
            raise OCRError('Solver returned error %s', result)
        return tuple(result)

    def report_code(self, cid, status):
        def accept(that, reqid, seqnum, status, data):
            if status == wzrpc.status.success:
                self.log.debug('Successfully reported captcha status')
            elif status == wzrpc.status.error:
                self.log.error('Solver returned error on report: %s', repr(data))
            elif status == wzrpc.status.e_req_denied:
                self.log.warn('Status {0}, reauthentificating'.\
                    format(wzrpc.name_status(status)))
                self.p.auth_requests()
            else:
                self.log.warn('Status {0}, retrying'.format(wzrpc.name_status(status)))
                that.retry = True
        self.p.wz_wait_reply(accept,
            b'Solver', b'report', (status.encode('utf-8'), cid.encode('utf-8')))
                
    def __call__(self, parent):
        self.p = parent
        self.log = parent.log
        self.running = parent.running
        self.sleep = parent.inter_sleep
        self.p.wz_auth_requests = [
            (b'Evaluator', b'evaluate'),
            (b'Solver', b'solve'),
            (b'Solver', b'report')]
        cst = cstate(self, WTState.starting)
        cst.__enter__()
        self.p.sig_sock.setsockopt(zmq.SUBSCRIBE, b'WipeThread')
        def handle_lsn(interface, method, data):
            if hasattr(self, 'spawn') and self.spawn:
                self.log.info('My current spawn is %s, state %s',
                    self.spawn.name, self.spawn.state.name)
            else:
                self.log.debug('Currently I do not have spawn')
        self.p.wz.set_sig_handler(b'WipeThread', b'log-spawn-name', handle_lsn)
        def handle_te(interface, method, data):
            if self.state is WTState.empty:
                self.p.term()
        self.p.wz.set_sig_handler(b'WipeThread', b'terminate-empty', handle_te)

        try:
            self.p.wz_connect()
            self.p.auth_requests()
        except WorkerInterrupt as e:
            self.log.error(e)
            return
        with cstate(self, WTState.empty):
            while self.running.is_set():
                try:
                    self.spawn = self.spawnqueue.get(False)
                except Empty:
                    self.sleep(1)
                    continue
                with cstate(self, WTState.running):
                    try:
                        self.spawn.run(self)
                    except WorkerInterrupt as e:
                        self.log.error(e)
                    except Exception as e:
                        self.log.exception('Spawn throwed exception %s, requesting new', e)
                    del self.spawn
                    self.spawn = None
                    self.spawnqueue.task_done()
        cst.__exit__(None, None, None)

class WipeSkel(object):
    reglimit = 10
    loglimit = 10
    conlimit = 3
    catrymax = 3
    _capdata = (0, 0)
    caprate = 0
    caprate_minp = 10
    caprate_limit = 0.9
    successtimeout = 1
    comment_successtimeout = 0
    topic_successtimeout = 0.8
    counter_report_interval = 60
    errortimeout = 3
    uqtimeout = 5  # Timeout for userqueue
    stoponclose = True
    die_on_neterror = False
    def __init__(self, pc, rp, domain, mrc, userqueue=None):
        self.pc = pc
        self.rp = rp
        self.state = WipeState.null
        self.site = beon.Beon(domain, self.http_request)
        self.name = '.'.join((
            type(self).__name__,
            self.rp.proxy.replace('.', '_') if self.rp.proxy
            else 'noproxy',
            self.site.domain.replace('.', '_')))
        self.rp.default_encoding = 'cp1251'
        self.rp.default_decoding = 'cp1251'
        self.rp.def_referer = self.site.ref  # Referer for net.py
        self.hooks = {
            'pre_register_new_user': [],
            'post_register_new_user': [],
            'post_login': [],
            'check_new_user': [],
        }
        self.counter_ticker = Ticker()
        self.counters = Counter()
        self.task_deque = deque()
        self.logined = False
        self.noproxy_rp = self.pc.noproxy_rp
        self.mrc = mrc
        if userqueue:
            self.userqueue = userqueue
        else:
            self.userqueue = Queue()

    def schedule(self, task, args=(), kvargs={}):
        self.task_deque.appendleft((task, args, kvargs))

    def schedule_first(self, task, args=(), kvargs={}):
        self.task_deque.append((task, args, kvargs))

    def perform_tasks(self):
        with cstate(self, WipeState.running):
            while self.w.running.is_set():
                self.counter_tick()
                try:
                    t = self.task_deque.pop()
                except IndexError:
                    return
                t[0](*t[1], **t[2])

    def long_sleep(self, time):
        time = int(time)
        with cstate(self, WipeState.sleeping):
            step = int(time/10 if time > 10 else 1)
            for s in range(0, time, step):
                self.w.sleep(step)
                self.counter_tick()

    def http_request(self, url, postdata=None, onlyjar=False, referer=None,
                     encoding=None, decoding=None):
        _conc = 0
        while self.w.running.is_set():
            _conc += 1
            try:
                return self.rp.http_req(
                    url, postdata, onlyjar, referer, encoding, decoding)
            except sup.NetError as e:
                if isinstance(e, sup.ConnError):
                    if self.die_on_neterror and _conc > self.conlimit:
                        raise
                    self.log.warn('%s, waiting. t: %s', e.args[0], _conc)
                    self.w.sleep(self.errortimeout)
                else:
                    self.log.error('%d %s', e.ec, e.args[0])
                    if self.die_on_neterror:
                        raise
                    else:
                        self.w.sleep(10)
        else:
            raise WorkerInterrupt()

    def gen_userdata(self):
        return short_wordsgen()

    def update_caprate(self, got):
        p, g = self._capdata
        p += 1
        if got is True:
            self.counters['captchas'] += 1
            g += 1
        if p >= 255:
            p = p/2
            g = g/2
        self._capdata = (p, g)
        self.caprate = g/p
        self.log.debug('Caprate: pos:%f got:%f rate:%f',
                       p, g, self.caprate)
        if (self.caprate_limit > 0
            and p > self.caprate_minp
            and self.caprate > self.caprate_limit):
            self.on_caprate_limit(self.caprate)
            # if self.getuser() == 'guest':
            #     self.log.info("lol, we were trying to post from guest")
            #     while not self.relogin(): self.w.sleep(self.errortimeout)
            # else:
            #     while not self.dologin(): self.w.sleep(self.errortimeout)

    def counter_tick(self):
        if self.counter_report_interval == 0:
            return
        e = self.counter_ticker.elapsed(False)
        if e > self.counter_report_interval:
            self.counter_ticker.tick()
            ccount = self.counters['comments']
            tcount = self.counters['topics']
            if ccount > 0:
                self.log.info('%d comments in %d seconds, %0.2f cps, %0.2f caprate',
                    ccount, e, ccount/e, self.caprate)
                self.counters['comments'] = 0
            if tcount > 0:
                self.log.info('%d topics in %d seconds, %0.2f tps, %0.2f caprate',
                    tcount, e, tcount/e, self.caprate)
                self.counters['topics'] = 0

    def on_caprate_limit(self, rate):
        if not self.logined:
            self._capdata = (0, 0)
            return
        self.log.warn('Caprate %f is over the limit', rate)
        raise Exception('Caprate limit reached')

    def captcha_wrapper(self, inc_fun, fin_fun, *args, **kvargs):
        # TODO: report codes after solving cycle instead of scheduling them.
        try:
            self.log.debug('captcha_wrapper: calling inc_fun %s', repr(inc_fun))
            self.log.error('captcha_wrapper: inc_fun returned %s',
                           repr(inc_fun(*args, **kvargs)))
        except beon.Success as e:
            self.update_caprate(False)
            raise
        except beon.Captcha as e:
            self.log.warn(e)
            _page = e.page
            _catry = e.catry
            # Don't update caprate with positives if not logined
            if self.logined is True:
                try:
                    user = self.find_login(_page)
                except beon.PermanentError:
                    self.log.debug(e)
                else:
                    if user != self.site.ud['login']:
                        self.log.warn('We were posting as %s, but our login is %s',
                                      user, self.site.ud['login'])
                        self.schedule_first(self.relogin)
                        return
            self.update_caprate(True)
            reports = []
            def r():
                if len(reports) > 0:
                    with cstate(self, WipeState.reporting_code):
                        for cid, status in reports:
                            self.report_code(cid, status)
                    reports.clear()
            while self.w.running.is_set():
                _requested_new = False
                try:
                    with cstate(self, WipeState.solving_captcha):
                        cahash, cacode, cid = self.solve_captcha(_page)
                except TempOCRError as e:
                    self.log.error('OCRError: %s, retrying', e)
                    self.w.sleep(self.errortimeout)
                    continue
                except OCRError as e:
                    self.log.error('OCRError: %s, requesting new captcha', e)
                    _requested_new = True
                    cahash, cacode, cid = e.cahash, '', None
                else:
                    self.log.info('code: %s', cacode)
                try:
                    self.log.debug('captcha_wrapper calling fin_fun %s', repr(fin_fun))
                    self.log.error('captcha_wrapper: fin_fun returned %s',
                        repr(fin_fun(cahash, cacode, *args, catry=_catry, **kvargs)))
                    break
                except beon.Success as e:
                    self.counters['captchas_solved'] += 1
                    if cid:
                        reports.append((cid, 'good'))
                    r()
                    raise
                except beon.Captcha as e:
                    _catry = e.catry
                    _page = e.page
                    if _requested_new:
                        self.log.warn('New captcha requested c:%d', _catry)
                        continue
                    self.log.warn('%s c:%d', e, _catry)
                    self.counters['captchas_wrong'] += 1
                    if cid:
                        reports.append((cid, 'bad'))
                    if _catry > self.catrymax:
                        r()
                        raise
                except Exception as e:
                    if cid:
                        reports.append((cid, 'bad'))
                    r()
                    raise

    def adaptive_timeout_wrapper(self, fun, *args, **kvargs):
        try:
            return fun(*args, **kvargs)
        except beon.Antispam as e:
            self.log.info('Antispam exc caught, successtimeout + 0.1, cur: %f',
                          self.successtimeout)
            self.successtimeout = self.successtimeout + 0.1
            raise

    def register_new_user(self):
        with cstate(self, WipeState.registering):
            _regcount = 0
            while self.w.running.is_set():
                <vul/>self.w.p.poll()</vul>
                ud = self.gen_userdata()
                self.request_email(ud)
                for c in self.hooks['pre_register_new_user']:
                    c(self, ud)
                self.log.info('Generated new userdata: %s, registering', ud['login'])
                self.log.debug('Userdata: %s', repr(ud))
                try:
                    udc = ud.copy()
                    if 0 in udc:
                        del udc[0]
                    self.register(**udc)
                except beon.Success as e:
                    self.validate_email(ud)
                    for c in self.hooks['post_register_new_user']:
                        c(self, ud)
                    return ud
                except (beon.EmptyAnswer, beon.Wait5Min) as e:
                    self.log.error('%s, sleeping for 100 seconds', e)
                    self.long_sleep(100)
                except beon.Captcha as e:
                    self.log.error('Too much wrong answers to CAPTCHA')
                    continue
                except beon.UnknownAnswer as e:
                    _regcount += 1
                    if not _regcount < self.reglimit:
                        raise beon.RegRetryLimit('Cannot register new user')
                    self.log.error('%s, userdata may be invalid, retrying c:%d',
                                e, _regcount)
                    self.w.sleep(self.errortimeout)
            else:
                raise WorkerInterrupt()

    def get_new_user(self):
        ud = self.userqueue.get(True, self.uqtimeout)
        self.userqueue.task_done()
        for c in self.hooks['check_new_user']:
            c(self, ud)
        return ud

    def login(self, login, passwd, **kvargs):
        if not self.site.login_lock.acquire(False):
            with self.site.login_lock.acquire():
                return
        self.logined = False
        try:
            self.captcha_wrapper(self.site.logininc, self.site.loginfin,
                                 login, passwd, **kvargs)
        except beon.Success as e:
            self.logined = True
            self.counters['logged_in'] += 1
            self.log.info(e)
            raise
        finally:
            self.site.login_lock.release()

    def find_login(self, rec):
        try:
            return re.findall(regexp.var_login, rec)[0]
        except IndexError:
            raise beon.PermanentError('No users in here')

    def get_current_login(self):
        return self.find_login(self.site.get_page('1'))

    def dologin(self):
        '''Choose user, do login and return it.'''
        while self.w.running.is_set():
            try:
                self.site.ud = self.get_new_user()
            except Empty:
                self.log.info('No users in queue')
                self.site.ud = self.register_new_user()
                return
            try:
                with cstate(self, WipeState.logging_in):
                    self.login(self.site.ud['login'], self.site.ud['passwd'])
            except beon.Success as e:
                self.site.postuser = self.site.ud['login']
                self.site.postpass = self.site.ud['passwd']
                self.validate_email(self.site.ud)
                for c in self.hooks['post_login']:
                    c(self, self.site.ud)
                self.w.sleep(self.successtimeout)
                return
            except beon.Captcha as e:
                self.log.error('Too many wrong answers to CAPTCHA')
                self.schedule(self.long_sleep, (10,))
                self.schedule(self.dologin)
            except beon.InvalidLogin as e:
                self.log.error("Invalid login, passing here")
                self.schedule(self.dologin)
                self.w.sleep(self.errortimeout)
            except beon.TemporaryError as e:
                self.userqueue.put(self.site.ud)
                self.log.warn(e)
                self.schedule(self.dologin)
                self.w.sleep(self.errortimeout)
        # else:
        #     pending = len(self.pc.sets['pending'])
        #     self.log.warn("No more logins here, %s pending."%pending)
        #     if pending == 0: return False

    def relogin(self):
        '''Relogin with current user or do login'''
        if 'login' in self.site.ud:
            while self.w.running.is_set():
                try:
                    with cstate(self, WipeState.logging_in):
                        self.login(self.site.ud['login'], self.site.ud['passwd'])
                except beon.Success as e:
                    for c in self.hooks['post_login']:
                        c(self, self.site.ud)
                    self.w.sleep(self.successtimeout)
                    return
                except beon.InvalidLogin as e:
                    self.log.error(e)
                    self.w.sleep(self.errortimeout)
                    break
                except beon.TemporaryError as e:
                    self.log.warn(e)
                    self.w.sleep(self.errortimeout)
                    continue
        self.dologin()

    def request_email(self, ud):
        ud['email'] = self.mailrequester.gen_addr()
        ud[0]['email_service'] = type(self.mailrequester).__name__
        ud[0]['email_requested'] = False
        ud[0]['email_validated'] = False

    def validate_email(self, ud):
        if ('email' not in ud or
            'email_service' not in ud[0] or
            'email_requested' not in ud[0] or
            'email_validated' not in ud[0] or
            not ud[0]['email_service'] == type(self.mailrequester).__name__
            or ud[0]['email_validated'] is True):
            return
        if not ud[0]['email_requested']:
            try:
                self.site.validate_email_inc()
            except beon.Success as e:
                ud[0]['email_requested'] = True
                self.log.info(e)
        self.log.info('Requesting messages for %s', ud['email'])
        messages = self.mailrequester.get_messages(ud['email'])
        for msg in messages:
            if not msg['mail_from'].find('<reminder@{0}>'.format(self.site.domain)):
                continue
            h = re.findall(regexp.hashinmail.format(self.site.domain),
                msg['mail_html'])
            if len(h) > 0:
                try:
                    self.site.validate_email_fin(h[0])
                except beon.Success as e:
                    ud[0]['email_validated'] = True
                    self.log.info(e)

    def switch_user(self):
        '''Log in with new user, but return the previous one'''
        if 'login' in self.site.ud:
            self.log.info('Switching user %s', self.site.ud['login'])
            self.return_user()
        self.site.ud = self.register_new_user()

    def return_user(self, ud=None):
        if not ud:
            if (hasattr(self.site, 'ud') and self.site.ud):
                ud = self.site.ud
                self.site.ud = None
            else:
                return
        self.log.info('Returning user %s to userqueue', ud['login'])
        self.userqueue.put(ud, False)

    def postmsg(self, target, msg, tuser=None, **kvargs):
        tpair = (tuser, target)
        target = target.lstrip('0')
        ptarget = (':'.join(tpair) if tuser else target)
        try:
            try:
                self.site.ajax_addcomment(target, msg, tuser, **kvargs)
            except beon.Success as e:
                self.update_caprate(False)
                raise
            except beon.Redir as e:
                self.log.warn(e)
                self.log.warn('Using non-ajax addcomment')
                self.captcha_wrapper(self.site.addcomment, self.site.addcommentfin,
                                     target, msg, tuser, **kvargs)
        except beon.Success as e:
            self.counters['comments_added'] += 1
            self.log.debug(e)
            raise
        except beon.Antispam as e:
            self.counters['antispam'] += 1
            self.comment_successtimeout = self.comment_successtimeout + 0.1
            self.log.info('Antispam exc caught, comment_successtimeout + 0.1, cur: %f',
                self.comment_successtimeout)
            raise
        except beon.GuestDeny as e:
            self.counters['delogin'] += 1
            self.log.warn('%s, trying to log in', e)
            self.schedule_first(self.relogin)
            raise
        except beon.Bumplimit as e:
            self.log.info(e)
            self.pc.sets['bumplimit'].add(tpair)
            raise
        except (beon.Closed, beon.UserDeny) as e:
            self.pc.sets['closed'].add(tpair)
            if self.stoponclose:
                self.log.info(e)
                raise beon.PermClosed("%s:%s is closed", tpair, e.answer)
            else:
                self.log.info('%s, starting 300s remove timer', e)
                self.pc.add_waiting('closed', tpair, 300)
                raise
        except beon.Wait5Min as e:
            self.counters['wait5mincount'] += 1
            self.log.warn(e)
            raise
        except beon.TemporaryError as e:
            self.log.warn(e)
            raise
        except beon.PermanentError as e:
            self.log.error(e)
            raise

    def addtopic(self, msg, subj, forum='1', tuser=None, **kvargs):
        try:
            self.captcha_wrapper(self.site.addtopicinc, self.site.addtopicfin,
                                 msg, forum, subj, tuser, **kvargs)
        except beon.Success as e:
            self.counters['topics_added'] += 1
            self.log.debug(e)
            raise
        except beon.Wait5Min as e:
            self.counters['wait5min'] += 1
            raise
            # self._bancount += 1
            # if 'login' in self.site.ud:
            #     self.log.warn(e)
            #     self.log.warn('Trying to change user')
            #     self.pc.sets['pending'].add(self.site.ud['login'])
            #     self.pc.add_waiting('pending', self.site.ud['login'], 300)
            #     self.dologin()
            # else:
            #     raise
        except beon.GuestDeny as e:
            if 'login' not in self.site.ud:
                raise
            self.counters['delogin'] += 1
            self.log.warn('%s, trying to log in', e)
            self.schedule_first(self.dologin)
            raise

    def register(self, login, passwd, name, email, **kvargs):
        self.logined = False
        try:
            self.captcha_wrapper(self.site.reginc, self.site.regfin,
                                 login, passwd, name, email, **kvargs)
        except beon.Success as e:
            self.log.info(e)
            self.logined = True
            self.counters['users_registered'] += 1
            raise

    def solve_captcha(self, page):
        # with cstate(self, WipeState.deobfuscating_capage):
        self.log.info('Deobfuscating capage')
        capair = self.w.deobfuscate_capage(self.site.domain, page)
        self.log.info('Answer: %s', repr(capair))
        if len(capair) != 2:
            raise PermOCRError('Invalid answer from Evaluator')
        self.log.info('Downloading captcha image')
        try:
            img = self.http_request(capair[1])
        except sup.net.HTTPError as e:
            # check error code here
            self.log.error(e)
            raise PermOCRError('404 Not Found on caurl', cahash=capair[0])
        self.log.info('Sending captcha image to solver')
        try:
            result, cid = self.w.solve_captcha(img)
        except OCRError as e:
            e.cahash = capair[0]
            raise
        return capair[0], result, cid
    
    def report_code(self, cid, status):
        self.log.info('Reporting %s code for %s', status, cid)
        self.w.report_code(cid, status)
        self.counters['captcha_codes_reported'] += 1

    def run(self, caller):
        self.w = caller
        self.log = logging.getLogger(self.name)
        self.run_time = Ticker()
        cst = cstate(self, WipeState.starting)
        cst.__enter__()
        self.mailrequester = self.mrc(self.noproxy_rp, self.w.running, self.w.sleep)

        # Get our own logger here, or use worker's?
        self.log.info('Starting')
        self.run_time.tick()

        def drop_user_handler(interface, method, data):
            self.log.info('drop-user signal recieved')
            self.dologin()

        self.w.p.wz.set_sig_handler(b'WipeSkel', b'drop-user', drop_user_handler)
        self.w.p.sig_sock.setsockopt(zmq.SUBSCRIBE, b'WipeSkel')

        try:
            self._run()
        except Exception as e:
            self.log.exception(e)
        cst.__exit__(None, None, None)
        with cstate(self, WipeState.terminating):
            self.w.p.sig_sock.setsockopt(zmq.UNSUBSCRIBE, b'WipeSkel')
            self.w.p.wz.del_sig_handler(b'WipeSkel', b'drop-user')
            self.log.info(repr(self.counters))
        self.log.info('Terminating, runtime is %ds', self.run_time.elapsed(False))
