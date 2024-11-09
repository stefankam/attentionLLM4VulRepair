#!/usr/bin/python
import sys
import sqlite3 as sqlite
import datetime

class na:
    def __init__(self, filename="na.db", authorId=1, debug=0):
        con = sqlite.connect(filename)
        if not con:
            print "error opening connection"
            sys.exit(1)
        self.con = con
        self.cur = con.cursor()
        self.authorId = authorId
        self.debug = debug 

    def add(self, title="", keywords="", content="", privacy=0):
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d %H:%M:%S")
        <fix/>self.cur.execute("INSERT INTO note(authorId, date, title, content, privacy) VALUES(?, ?, ?, ?, ?);",</fix>
                (self.authorId, date, title, content, privacy))
        noteId = self.cur.lastrowid
        for keyword in keywords:
            <fix/>if self.debug:
                print " inserting keyword:", keyword
            <fix/>keywordId = self.con.execute("SELECT keywordId FROM keyword WHERE keyword = ?;", [keyword]).fetchone()</fix></fix>
            if keywordId:
                if self.debug:
                    print "(existing keyword with id:", keywordId, ")"
                keywordId = keywordId[0]
            else:
                <fix/>if self.debug:
                    print "(new keyword)"
                self.con.execute("INSERT INTO keyword(keyword) VALUES (?);", [keyword])</fix>
                keywordId = self.cur.lastrowid
            <fix/>self.con.execute("INSERT INTO notekeyword(noteId, keywordID) VALUES(?, ?)", [noteId, keywordId])</fix>
        self.con.commit()
        return(noteId)
   
    def find(self, keywords=""):
        noteIds = []
        if keywords[0] == "?":
            <fix/>noteIds.extend(self.con.execute("SELECT noteId FROM note;"))</fix>
        else:
            for keyword in keywords:
                if self.debug:
                    print "keyword:", keyword, "..."
                keywordId = self.cur.execute("SELECT keywordId FROM keyword WHERE keyword = ?;", [keyword])
                try:
                    keywordId = self.con.execute("SELECT keywordId FROM keyword WHERE keyword = ?;", [keyword]).fetchone()
                    if keywordId:
                        <fix/>for noteId in self.cur.execute("SELECT noteId FROM notekeyword WHERE keywordId = ?;", keywordId):</fix>
                            if self.debug:
                                <fix/>print '  noteId:', noteId</fix>
                            if noteId not in noteIds:
                                noteIds.append(noteId)
                except:
                    print "problem"
                    pass
        for n in noteIds:
            <fix/>res = self.cur.execute("SELECT noteId, date, title, content, privacy FROM note WHERE noteId=?;", n).fetchone()</fix>
            if int(res[4]) > 0:
                privacy = "(Private)"
            else:
                privacy = "(Public)"
            print "<%s %s> %s %s\n  %s" % (res[0], res[1], res[2], privacy, res[3])
            <fix/>keywordIds = []
            keywordIds.extend(self.con.execute("SELECT keywordid FROM notekeyword WHERE notekeyword.noteid = ?;", n))</fix>
            keywords = []
            <fix/>for k in keywordIds:
                keywords.append(self.cur.execute("SELECT keyword FROM keyword WHERE keywordId = ?;", k).fetchone()[0])</fix>
            print " ", ", ".join(keywords[i] for i in range(len(keywords))), "\n"

 
