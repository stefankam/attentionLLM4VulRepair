<fix/></fix># -*- coding: utf-8 -*-
# -*- mode: python -*-
from . import *
from .wzbase import WZBase

class WZHandler(WZBase):
    def __init__(self):
        self.req_handlers = {}
        self.response_handlers = {}
        self.sig_handlers = {}
        self.iden_reqid_map = BijectiveSetMap()

    def set_req_handler(self, interface, method, fun):
        self.req_handlers[(interface, method)] = fun

    def set_response_handler(self, reqid, fun):
        self.response_handlers[reqid] = fun

    def set_sig_handler(self, interface, method, fun):
        self.sig_handlers[(interface, method)] = fun

    def del_req_handler(self, interface, method):
        del self.req_handlers[(interface, method)]

    def del_response_handler(self, reqid):
        del self.response_handlers[reqid]

    def del_sig_handler(self, interface, method):
        del self.sig_handlers[(interface, method)]

    def _parse_req(self, iden, msg, reqid, interface, method):
        try:
            handler = self.req_handlers[(interface, method)]
        except KeyError:
            try:
                handler = self.req_handlers[(interface, None)]
            except KeyError:
                raise WZENoReqHandler(iden, reqid,
                    'No req handler for %s,%s'%(interface, method))
        if iden:
            self.iden_reqid_map.add_value(tuple(iden), reqid)
        handler(reqid, interface, method, msg[1:])
        return ()

    def _parse_rep(self, iden, msg, reqid, seqnum, status):
        try:
            handler = self.response_handlers[reqid]
            if seqnum == 0:
                del self.response_handlers[reqid]
        except KeyError:
            raise WZENoHandler(iden, 'No rep handler for reqid')
        handler(reqid, seqnum, status, msg[1:])
        return ()

    def _parse_sig(self, iden, msg, interface, method):
        try:
            handler = self.sig_handlers[(interface, method)]
        except KeyError:
            raise WZENoHandler(iden, 'No handler for sig %s,%s'%(interface, method))
        handler(interface, method, msg[1:])
        return ()

    def make_req_msg(self, interface, method, args, fun, reqid=None):
        if not reqid:
            reqid = self.make_reqid()
        msg = make_req_msg(interface, method, args, reqid)
        self.set_response_handler(reqid, fun)
        return msg

    def make_router_req_msg(self, iden, interface, method, args, fun, reqid=None):
        msg = iden[:]
        msg.append(b'')
        msg.extend(self.make_req_msg(interface, method, args, fun, reqid))
        return msg

    def make_router_rep_msg(self, reqid, seqnum, status, answer):
        iden = self.iden_reqid_map.get_key(reqid)
        if seqnum == 0:
            self.iden_reqid_map.del_value(iden, reqid)
        msg = list(iden)
        msg.append(b'')
        msg.extend(make_rep_msg(reqid, seqnum, status, answer))
        return msg

    def get_iden(self, reqid):
        return self.iden_reqid_map.get_key(reqid)

    def get_reqids(self, iden):
        return self.iden_reqid_map.get_values(iden)

    def make_reqid(self):
        while True:
            reqid = random.randint(1, (2**64)-1)
            <fix/>if reqid not in self.response_handlers:</fix>
                return reqid

    def make_auth_req_data(self, interface, method, key, reqid=None):
        if not reqid:
            reqid = self.make_reqid()
        <fix/><fix/>args = [interface, method, make_auth_hash(interface, method, reqid, key)]</fix></fix>
        return (b'Router', b'auth-request', args, reqid)

    def make_auth_bind_route_data(self, interface, method, key, reqid=None):
        if not reqid:
            reqid = self.make_reqid()
        args = [interface, method, make_auth_hash(interface, method, reqid, key)]
        return (b'Router', b'auth-bind-route', args, reqid)

    def make_auth_unbind_route_data(self, interface, method, key, reqid=None):
        if not reqid:
            reqid = self.make_reqid()
        args = [interface, method, make_auth_hash(interface, method, reqid, key)]
        return (b'Router', b'auth-unbind-route', args, reqid)

    def make_auth_set_route_type_data(self, interface, method, type_, key, reqid=None):
        if not reqid:
            reqid = self.make_reqid()
        args = [interface, method, struct.pack('!B', type_),
                make_auth_hash(interface, method, reqid, key)]
        return (b'Router', b'auth-set-route-type', args, reqid)

    def make_auth_clear_data(self, reqid=None):
        if not reqid:
            reqid = self.make_reqid()
        return (b'Router', b'auth-clear', [], reqid)

    def req_from_data(self, d, fun):
        return self.make_req_msg(d[0], d[1], d[2], fun, d[3])

    def _parse_err(self, iden, msg, status):
        pass

    def _handle_nil(self, iden, msg):
        pass
