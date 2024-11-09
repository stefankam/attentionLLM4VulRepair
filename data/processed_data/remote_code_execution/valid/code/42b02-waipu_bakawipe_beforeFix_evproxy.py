# -*- coding: utf-8 -*-
# -*- mode: python -*-
import wzrpc
from sup.ticker import Ticker

class EvaluatorProxy:
    def __init__(self, ev_init, *args, **kvargs):
        super().__init__()
        self.ev_init = ev_init
        self.bind_kt_ticker = Ticker()
        self.bind_kt = 5

    def handle_evaluate(self, reqid, interface, method, data):
        domain, page = data
        self.p.log.info('Recvd page %s, working on', reqid)
        res = self.ev.solve_capage(domain, page)
        self.p.log.info('Done, sending answer: %s', res)
        self.p.send_success_rep(reqid, [v.encode('utf-8') for v in res])

    def send_keepalive(self):
        msg = self.p.wz.make_req_msg(b'Router', b'bind-keepalive', [],
            self.handle_keepalive_reply)
        msg.insert(0, b'')
        self.p.wz_sock.send_multipart(msg)

    def handle_keepalive_reply(self, reqid, seqnum, status, data):
        if status == wzrpc.status.success:
            self.p.log.debug('Keepalive was successfull')
        elif status == wzrpc.status.e_req_denied:
            self.p.log.warn('Keepalive status {0}, reauthentificating and rebinding'.
                format(wzrpc.name_status(status)))
            self.p.auth_requests()
            self.p.bind_methods()
        elif status == wzrpc.status.e_timeout:
            self.p.log.warn('Keepalive timeout')
        else:
            self.p.log.warn('Keepalive status {0}'.
                format(wzrpc.name_status(status)))

    def __call__(self, parent):
        self.p = parent
        self.p.wz_connect()
        self.p.wz_auth_requests = [
            (b'Router', b'auth-bind-route'),
            (b'Router', b'auth-unbind-route'),
            (b'Router', b'auth-set-route-type')]
        self.p.wz_bind_methods = [
            (b'Evaluator', b'evaluate', self.handle_evaluate, wzrpc.routetype.random)]
        self.p.auth_requests()
        self.p.bind_methods()
        self.ev = self.ev_init()
        self.bind_kt_ticker.tick()
        while self.p.running.is_set():
            <vul/>socks = self.p.poll()</vul>
            if self.bind_kt_ticker.elapsed(False) > self.bind_kt:
                self.bind_kt_ticker.tick()
                self.send_keepalive()
