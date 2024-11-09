# -*- coding: utf-8 -*-
# -*- mode: python -*-
from sup.net import NetError
from wzworkers import WorkerInterrupt
from wipeskel import WipeSkel, WipeState, cstate
from beon import exc, regexp
from collections import ChainMap
import re

class UniWipe(WipeSkel):
    def __init__(self, forums, targets, sbjfun, msgfun, *args, **kvargs):
        self.sbjfun = sbjfun
        self.msgfun = msgfun
        self.forums = forums
        self.targets = (type(targets) == str and [('', targets)]
                        or type(targets) == tuple and list(targets)
                        or targets)
        super().__init__(*args, **kvargs)
        self.ignore_map = ChainMap(
            self.pc.sets['closed'], self.pc.sets['bumplimit'],
            self.pc.sets['bugged'], self.pc.sets['protected'],
            self.targets)

    def on_caprate_limit(self, rate):
        if not self.logined:
            self._capdata = (0, 0)
            return
        self.log.warning('Caprate limit reached, calling dologin() for now')
        self.dologin()
        # super().on_caprate_limit(rate)

    def comment_loop(self):
        for t in self.targets:
            self.schedule(self.add_comment, (t, self.msgfun()))
        if len(self.targets) == 0:
            self.schedule(self.scan_targets_loop)
        else:
            self.schedule(self.comment_loop)

    def add_comment(self, t, msg):
        # with cstate(self, WipeState.posting_comment):
        if True: # Just a placeholder
            try:
                # self.counter_tick()
                self.postmsg(t[1], msg, t[0])
            except exc.Success as e:
                self.counters['comments'] += 1
                self.w.sleep(self.comment_successtimeout)
            except exc.Antispam as e:
                self.w.sleep(self.comment_successtimeout)
                self.schedule(self.add_comment, (t, msg))
            except (exc.Closed, exc.UserDeny) as e:
                try:
                    self.targets.remove(t)
                except ValueError:
                    pass
                self.w.sleep(self.comment_successtimeout)
            except exc.Captcha as e:
                self.log.error('Too many wrong answers to CAPTCHA')
                self.schedule(self.add_comment, (t, msg))
            except exc.UnknownAnswer as e:
                self.log.warn('%s: %s', e, e.answer)
                self.schedule(self.add_comment, (t, msg))
            except exc.Wait5Min as e:
                self.schedule(self.add_comment, (t, msg))
                self.schedule_first(self.switch_user)
            except exc.EmptyAnswer as e:
                <fix/>self.log.info('Removing %s from targets and adding to bugged', t)
                self.pc.sets['bugged'].add(t)
                try:
                    self.targets.remove(t)
                except ValueError as e:
                    pass
                self.w.sleep(self.errortimeout)
            except exc.TopicDoesNotExist as e:
                self.log.info('Removing %s from targets and adding to bugged', t)
                self.pc.sets['bugged'].add(t)</fix>
                try:
                    self.targets.remove(t)
                except ValueError as e:
                    pass
                self.w.sleep(self.errortimeout)
            except exc.TemporaryError as e:
                self.schedule(self.add_comment, (t, msg))
                self.w.sleep(self.errortimeout)
            except exc.PermanentError as e:
                try:
                    self.targets.remove(t)
                except ValueError as e:
                    pass
                self.w.sleep(self.errortimeout)
            except UnicodeDecodeError as e:
                self.log.exception(e)
                self.w.sleep(self.errortimeout)

    def forumwipe_loop(self):
        <fix/>for f in self.forums.copy():</fix>
            self.counter_tick()
            try:
                self.addtopic(self.msgfun(), self.sbjfun(), f)
            except exc.Success as e:
                self.counters['topics'] += 1
                self.w.sleep(self.topic_successtimeout)
            except exc.Wait5Min as e:
                self.topic_successtimeout = self.topic_successtimeout + 0.1
                self.log.info('Wait5Min exc caught, topic_successtimeout + 0.1, cur: %f',
                    self.topic_successtimeout)
                self.w.sleep(self.topic_successtimeout)
            except exc.Captcha as e:
                self.log.error('Too many wrong answers to CAPTCHA')
                self.long_sleep(10)
            except exc.UnknownAnswer as e:
                self.log.warning('%s: %s', e, e.answer)
                self.w.sleep(self.errortimeout)
            except exc.PermanentError as e:
                self.log.error(e)
                self.w.sleep(self.errortimeout)
            except exc.TemporaryError as e:
                self.log.warn(e)
                self.w.sleep(self.errortimeout)

    def get_targets(self):
        found_count = 0
        for user, forum in self.forums:
            targets = []
            self.log.debug('Scanning first page of the forum %s:%s', user, forum)
            page = self.site.get_page('1', forum, user)
            rxp = re.compile(regexp.f_sub_id.format(user, self.site.domain, forum))
            found = set(map(lambda x: (user, x[0]+x[1]), rxp.findall(page)))
            for t in found:
                <fix/>if t in self.ignore_map:</fix>
                    continue
                targets.append(t)
            lt = len(targets)
            found_count += lt
            if lt > 0:
                self.log.info('Found %d new targets in forum %s:%s', lt, user, forum)
            else:
                self.log.debug('Found no new targets in forum %s:%s', user, forum)
            self.targets.extend(targets)
        return found_count

    def scan_targets_loop(self):
        with cstate(self, WipeState.scanning_for_targets):
            while len(self.targets) == 0:
                c = self.get_targets()
                if c == 0:
                    self.log.info('No targets found at all, sleeping for 30 seconds')
                    self.long_sleep(30)
            self.schedule(self.comment_loop)
        if len(self.forums) == 0:
            self.schedule(self.wait_loop)

    def wait_loop(self):
        if len(self.targets) > 0:
            self.schedule(self.comment_loop)
            return
        if len(self.forums) == 0:
            with cstate(self, WipeState.waiting_for_targets):
                while len(self.forums) == 0:
                    # To prevent a busy loop.
                    self.counter_tick()
                    self.w.sleep(1)
        self.schedule(self.scan_targets_loop)

    def _run(self):
        self.schedule(self.dologin)
        self.schedule(self.wait_loop)
        self.schedule(self.counter_ticker.tick)
        try:
            self.perform_tasks()
        except NetError as e:
            self.log.error(e)
        except WorkerInterrupt as e:
            self.log.warning(e)
        except Exception as e:
            self.log.exception(e)
        self.return_user()
# tw_flag = False
# if len(self.targets) > 0:
#     with cstate(self, WipeState.posting_comment):
#         while len(self.targets) > 0:
#             self.threadwipe_loop()
#     if not tw_flag:
#         tw_flag = True
# if tw_flag:
#     # Sleep for topic_successtimeout after last comment
#     # to prevent a timeout spike
#     self.w.sleep(self.topic_successtimeout)
#     tw_flag = False
# with cstate(self, WipeState.posting_topic):
# self.forumwipe_loop()
