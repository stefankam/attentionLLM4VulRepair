<fix/></fix>import zmq
import threading, multiprocessing
import logging
from sup.ticker import Ticker
# from sup import split_frames
import wzrpc
import exceptions
from wzrpc.wzhandler import WZHandler
import wzauth_data

class WorkerInterrupt(Exception):
    '''Exception to raise when self.running is cleared'''
    def __init__(self):
        super().__init__('Worker was interrupted at runtime')

class Suspend(Exception):
    # if we need this at all.
    '''Exception to raise on suspend signal'''
    def __init__(self, interval, *args, **kvargs):
        self.interval = interval
        super().__init__(*args, **kvargs)

class Resume(Exception):
    '''Exception to raise when suspend sleep is interrupted'''

class WZWorkerBase:
    def __init__(self, wz_addr, fun, args=(), kvargs={},
            name=None, start_timer=None, poll_timeout=None,
            pargs=(), pkvargs={}):
        super().__init__(*pargs, **pkvargs)
        self.name = name if name else type(self).__name__
        self.start_timer = start_timer
        self.poll_timeout = poll_timeout if poll_timeout else 5*1000
        self.call = (fun, args, kvargs)

        self.wz_addr = wz_addr
        self.wz_auth_requests = []
        self.wz_bind_methods = []
        <fix/>self.wz_poll_timeout = 30 * 1000
        self.wz_retry_timeout = 5</fix>

    def __sinit__(self):
        '''Initializes thread-local interface on startup'''
        self.log = logging.getLogger(self.name)
        self.running = threading.Event()
        self.sleep_ticker = Ticker()
        self.poller = zmq.Poller()

        s = self.ctx.socket(zmq.SUB)
        self.poller.register(s, zmq.POLLIN)
        s.setsockopt(zmq.IPV6, True)
        s.connect(self.sig_addr)
        s.setsockopt(zmq.SUBSCRIBE, b'GLOBAL')
        s.setsockopt(zmq.SUBSCRIBE, b'WZWorker')
        s.setsockopt(zmq.SUBSCRIBE, bytes(self.name, 'utf-8'))
        self.sig_sock = s

        s = self.ctx.socket(zmq.DEALER)
        self.poller.register(s, zmq.POLLIN)
        s.setsockopt(zmq.IPV6, True)
        self.wz_sock = s

        self.wz = WZHandler()

        <fix/>def term_handler(i, m, d):</fix>
            self.log.info(
                'Termination signal %s recieved',
                <fix/>repr((i, m, d)))</fix>
            self.term()
            raise WorkerInterrupt()
        self.wz.set_sig_handler(b'WZWorker', b'terminate', term_handler)

        <fix/>def execute_handler(i, m, d):
            if len(d) < 1:
                return
            try:
                exec(d[0].decode('utf-8'))
            except Exception as e:
                self.log.exception(e)
        self.wz.set_sig_handler(b'WZWorker', b'execute', execute_handler)

        def suspend_handler(i, m, d):
            if len(d) != 1:
                self.log.waring('Suspend signal without a time recieved, ignoring')
            self.log.info('Suspend signal %s recieved', repr((i, m, d)))
            try:
                t = int(d[0])
                # raise Suspend(t)
                self.inter_sleep(t)
            except Resume as e:
                self.log.info(e)
            except Exception as e:
                self.log.error(e)
        self.wz.set_sig_handler(b'WZWorker', b'suspend', suspend_handler)

        def resume_handler(i, m, d):
            self.log.info('Resume signal %s recieved', repr((i, m, d)))</fix>
            raise Resume()
        self.wz.set_sig_handler(b'WZWorker', b'resume', resume_handler)

        self.running.set()

    def wz_connect(self):
        self.wz_sock.connect(self.wz_addr)

    def wz_wait_reply(self, fun, interface, method, data, reqid=None, timeout=None):
        <fix/><fix/>s, p, t = self.wz_sock, self.poll, self.sleep_ticker</fix></fix>
        timeout = timeout if timeout else self.wz_poll_timeout
        rs = wzrpc.RequestState(fun)
        msg = self.wz.make_req_msg(interface, method, data,
                                   rs.accept, reqid)
        msg.insert(0, b'')
        s.send_multipart(msg)
        t.tick()
        while self.running.is_set():
            p(timeout*1000)
            if rs.finished:
                if rs.retry:
                    self.inter_sleep(self.wz_retry_timeout)
                    msg = self.wz.make_req_msg(interface, method, data,
                        rs.accept, reqid)
                    msg.insert(0, b'')
                    s.send_multipart(msg)
                    rs.finished = False
                    rs.retry = False
                    continue
                return
            elapsed = t.elapsed(False)
            if elapsed >= timeout:
                t.tick()
                # Notify fun about the timeout
                rs.accept(None, 0, 255, [elapsed])
                # fun sets rs.retry = True if it wants to retry
        raise WorkerInterrupt()

    def wz_multiwait(self, requests):
        # TODO: rewrite the retry loop
        s, p, t = self.wz_sock, self.poll, self.sleep_ticker
        timeout = self.wz_poll_timeout
        rslist = []
        msgdict = {}
        for request in requests:
            rs = wzrpc.RequestState(request[0])
            rslist.append(rs)
            msg = self.wz.make_req_msg(request[1][0], request[1][1], request[1][2],
                                    rs.accept, request[1][3])
            msg.insert(0, b'')
            msgdict[rs] = msg
            s.send_multipart(msg)
        while self.running.is_set():
            flag = 0
            for rs in rslist:
                if rs.finished:
                    if not rs.retry:
                        del msgdict[rs]
                        continue
                    s.send_multipart(msgdict[rs])
                    rs.finished = False
                    rs.retry = False
                flag = 1
            if not flag:
                return
            # check rs before polling, since we don't want to notify finished one
            # about the timeout
            t.tick()
            p(timeout*1000)
            if t.elapsed(False) >= timeout:
                for rs in rslist:
                    if not rs.finished:
                        rs.accept(None, 0, 255, []) # Notify fun about the timeout
                        rs.finished = True # fun sets rs.retry = True if it wants to retry
        raise WorkerInterrupt()

    def auth_requests(self):
        for i, m in self.wz_auth_requests:
            def accept(that, reqid, seqnum, status, data):
                if status == wzrpc.status.success:
                    self.log.debug('Successfull auth for (%s, %s)', i, m)
                elif status == wzrpc.status.e_auth_wrong_hash:
                    <fix/>raise exceptions.PermanentError(</fix>
                        'Cannot authentificate for ({0}, {1}), {2}: {3}'.\
                        format(i, m, wzrpc.name_status(status), repr(data)))
                elif wzrpc.status.e_timeout:
                    self.log.warn('Timeout {0}, retrying'.format(data[0]))
                    that.retry = True
                else:
                    self.log.warning('Recvd unknown reply for (%s, %s) %s: %s', i, m,
                        wzrpc.name_status(status), repr(data))
            self.wz_wait_reply(accept,
                *self.wz.make_auth_req_data(i, m, wzauth_data.request[i, m]))


    def bind_route(self, i, m, f):
        self.log.debug('Binding %s,%s route', i, m)
        def accept(that, reqid, seqnum, status, data):
            if status == wzrpc.status.success:
                self.wz.set_req_handler(i, m, f)
                self.log.debug('Succesfully binded route (%s, %s)', i, m)
            elif status == wzrpc.status.e_req_denied:
                self.log.warn('Status {0}, reauthentificating'.\
                    format(wzrpc.name_status(status)))
                self.auth_requests()
            elif wzrpc.status.e_timeout:
                self.log.warn('Timeout {0}, retrying'.format(data[0]))
                that.retry = True
            else:
                self.log.warn('Status {0}, retrying'.format(wzrpc.name_status(status)))
                that.retry = True
        return self.wz_wait_reply(accept,
                *self.wz.make_auth_bind_route_data(i, m, wzauth_data.bind_route[i, m]))

    def set_route_type(self, i, m, t):
        self.log.debug('Setting %s,%s type to %d', i, m, t)
        def accept(that, reqid, seqnum, status, data):
            if status == wzrpc.status.success:
                self.log.debug('Succesfully set route type for (%s, %s) to %s', i, m,
                    wzrpc.name_route_type(t))
            elif status == wzrpc.status.e_req_denied:
                self.log.warn('Status {0}, reauthentificating'.\
                    format(wzrpc.name_status(status)))
                self.auth_requests()
            else:
                self.log.warn('Status {0}, retrying'.format(wzrpc.name_status(status)))
                that.retry = True
        return self.wz_wait_reply(accept,
            *self.wz.make_auth_set_route_type_data(i, m, t,
                wzauth_data.set_route_type[i, m]))

    def unbind_route(self, i, m):
        if not (i, m) in self.wz.req_handlers:
            self.log.debug('Route %s,%s was not bound', i, m)
            return
        self.log.debug('Unbinding route %s,%s', i, m)
        self.wz.del_req_handler(i, m)
        def accept(that, reqid, seqnum, status, data):
            if status == wzrpc.status.success:
                self.log.debug('Route unbinded for (%s, %s)', i, m)
            else:
                self.log.warn('Status %s, passing', wzrpc.name_status(status))
        return self.wz_wait_reply(accept,
            *self.wz.make_auth_unbind_route_data(i, m, wzauth_data.bind_route[i, m]))

    def clear_auth(self):
        self.log.debug('Clearing our auth records')
        def accept(that, reqid, seqnum, status, data):
            if status == wzrpc.status.success:
                self.log.debug('Auth records on router were cleared')
            else:
                self.log.warn('Status %s, passing', wzrpc.name_status(status))
        return self.wz_wait_reply(accept, *self.wz.make_auth_clear_data())

    def bind_methods(self):
        for i, m, f, t in self.wz_bind_methods:
            self.set_route_type(i, m, t)
            self.bind_route(i, m, f)

    def unbind_methods(self):
        for i, m, f, t in self.wz_bind_methods:
            self.unbind_route(i, m)
        <fix/># self.clear_auth()</fix>

    def send_rep(self, reqid, seqnum, status, data):
        self.wz_sock.send_multipart(
            self.wz.make_router_rep_msg(reqid, seqnum, status, data))

    def send_success_rep(self, reqid, data):
        self.send_rep(reqid, 0, wzrpc.status.success, data)

    def send_error_rep(self, reqid, data):
        self.send_rep(reqid, 0, wzrpc.status.error, data)

    def send_wz_error(self, reqid, data, seqid=0):
        msg = self.wz.make_dealer_rep_msg(
            reqid, seqid, wzrpc.status.error, data)
        self.wz_sock.send_multipart(msg)

    def send_to_router(self, msg):
        msg.insert(0, b'')
        self.wz_sock.send_multipart(msg)
    
    # def bind_sig_route(self, routetype, interface, method, fun):
    #     self.log.info('Binding %s,%s as type %d signal route',
    #                   interface, method, routetype)
    #     self.wz.set_signal_handler(interface, method, fun)
    #     msg = self.wz.make_dealer_sig_msg(b'Router', b'bind-sig-route',
    #                                       [interface, method],
    #                                       self.accept_ok)
    #     self.wz_sock.send_multipart(msg)

    # def unbind_sig_route(self, interface, method):
    #     self.log.info('Deleting %s,%s signal route', interface, method)
    #     self.wz.del_signal_handler(interface, method)
    #     msg = self.wz.make_dealer_sig_msg(b'Router', b'unbind-sig-route',
    #                                       [interface, method],
    #                                       self.accept_ok)
    #     self.wz_sock.send_multipart(msg)

    def inter_sleep(self, timeout):
        self.sleep_ticker.tick()
        while self.sleep_ticker.elapsed(False) < timeout:
            try:
                self.poll(timeout * 1000)
            <fix/>except Resume:</fix>
                return

    def poll(self, timeout=None):
        try:
            <fix/>socks = dict(self.poller.poll(timeout if timeout is not None</fix>
                else self.poll_timeout))
        except zmq.ZMQError as e:
            self.log.error(e)
            return
        if socks.get(self.sig_sock) == zmq.POLLIN:
            # No special handling or same-socket replies are necessary for signals.
            # Backwards socket replies may be added here.
            frames = self.sig_sock.recv_multipart()
            try:
                self.wz.parse_msg(frames[0], frames[1:])
            except wzrpc.WZError as e:
                self.log.warn(e)
        if socks.get(self.wz_sock) == zmq.POLLIN:
            self.process_wz_msg(self.wz_sock.recv_multipart())
        return socks

    def process_wz_msg(self, frames):
        try:
            for nfr in self.wz.parse_router_msg(frames):
                # Send replies from the handler, for cases when its methods were rewritten
                self.wz_sock.send_multipart(nfr)
        except wzrpc.WZErrorRep as e:
            self.log.info(e)
            self.wz_sock.send_multipart(e.rep_msg)
        except wzrpc.WZError as e:
            self.log.warn(e)

    def run(self):
        self.__sinit__()
        if self.start_timer:
            self.inter_sleep(self.start_timer)
        if self.running:
            self.log.info('Starting')
            try:
                self.child = self.call[0](*self.call[1], **self.call[2])
                self.child(self)
            except WorkerInterrupt as e:
                self.log.warn(e)
            except Exception as e:
                self.log.exception(e)
            self.log.info('Terminating')
        else:
            self.log.info('Aborted')
        self.running.set() # wz_multiwait needs this to avoid another state check.
        self.unbind_methods()
        self.running.clear()
        self.wz_sock.close()
        self.sig_sock.close()

    def term(self):
        self.running.clear()


class WZWorkerThread(WZWorkerBase, threading.Thread):
    def start(self, ctx, sig_addr, *args, **kvargs):
        self.ctx = ctx
        self.sig_addr = sig_addr
        threading.Thread.start(self, *args, **kvargs)

class WZWorkerProcess(WZWorkerBase, multiprocessing.Process):
    def start(self, sig_addr, *args, **kvargs):
        self.sig_addr = sig_addr
        multiprocessing.Process.start(self, *args, **kvargs)

    def __sinit__(self):
        self.ctx = zmq.Context()
        super().__sinit__()
