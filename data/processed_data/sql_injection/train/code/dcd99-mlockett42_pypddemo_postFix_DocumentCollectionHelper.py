#DocumentCollectionHelper.py
# A collection of functions that work with DocumentCollections and do thing not available in pyjs

import sqlite3
import os
from collections import defaultdict
from HistoryEdge import HistoryEdge
from HistoryGraph import HistoryGraph

def SaveDocumentCollection(dc, filenameedges, filenamedata):
    try:
        os.remove(filenameedges)
    except:
        pass
    c = sqlite3.connect(filenameedges)
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS edge (
                    documentid text, 
                    documentclassname text, 
                    edgeclassname text, 
                    edgeid text PRIMARY KEY, 
                    startnode1id text, 
                    startnode2id text, 
                    endnodeid text, 
                    propertyownerid text, 
                    propertyname text, 
                    propertyvalue text, 
                    propertytype text
                )''')
    c.execute("DELETE FROM edge")
    for documentid in dc.objects:
        documentlist = dc.objects[documentid]
        for document in documentlist:
            history = document.history
            edge = history.edges[edgeid]
            startnodes = list(edge.startnodes)
            if len(edge.startnodes) == 1:
                startnode1id = startnodes[0]
                startnode2id = ""
            elif len(edge.startnodes) == 2:
                startnode1id = startnodes[0]
                startnode2id = startnodes[1]
            else:
                assert False
            
            if edge.propertytype is None:
                propertytypename = ""
            else:
                propertytypename = edge.propertytype.__name__
            <fix/>c.execute("INSERT INTO edge VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (document.id, document.__class__.__name__, 
                edge.__class__.__name__, edge.edgeid, startnode1id, startnode2id, edge.endnode, edge.propertyownerid, edge.propertyname,
                edge.propertyvalue, propertytypename))
            #c.execute("INSERT INTO edge VALUES ('" + document.id + "', '" + document.__class__.__name__ + "', '" + edge.__class__.__name__ + "', '" + edge.edgeid + "', " +
            #    "'" + startnode1id + "', '" + startnode2id + "', '" + edge.endnode + "', '" + edge.propertyownerid + "', '" + edge.propertyname + "', '" + str(edge.propertyvalue) + "', "
            #    "'" + propertytypename + "')")</fix>

    c.commit()
    c.close()

    try:
        os.remove(filenamedata)
    except:
        pass
    database = sqlite3.connect(filenamedata)
    foreignkeydict = defaultdict(list)
    for classname in dc.classes:
        theclass = dc.classes[classname]
        variables = [a for a in dir(theclass) if not a.startswith('__') and not callable(getattr(theclass,a))]
        for a in variables:
            if isinstance(getattr(theclass, a), FieldList):
                foreignkeydict[getattr(theclass, a).theclass.__name__].append((classname, a))
    columndict = defaultdict(list)
    for classname in dc.classes:
        theclass = dc.classes[classname]
        variables = [a for a in dir(theclass) if not a.startswith('__') and not callable(getattr(theclass,a))]
        for a in variables:
            if isinstance(getattr(theclass, a), FieldList) == False:
                columndict[classname].append((a, "int" if isinstance(getattr(theclass, a), FieldInt) else "text"))
    for k in foreignkeydict:
        for (classname, a) in foreignkeydict[k]:
            columndict[k].append((classname + "id", "text"))
    for classname in columndict:
        columnlist = columndict[classname]
        sql = "CREATE TABLE " + classname + " (id text "
        for (a, thetype) in columnlist:
            sql += ","
            sql += a + " " + thetype
        sql += ")"

        database.execute(sql)
    
    for documentid in dc.objects:
        SaveDocumentObject(database, dc.objects[documentid][0], None, foreignkeydict, columndict)

    database.commit()

def SaveDocumentObject(self, documentobject, parentobject, foreignkeydict, columndict):
    variables = [a for a in dir(documentobject.__class__) if not a.startswith('__') and not callable(getattr(documentobject.__class__,a))]
    for a in variables:
        if isinstance(getattr(documentobject.__class__, a), FieldList):
            for childobj in getattr(documentobject, a):
                self.SaveDocumentObject(childobj, documentobject, foreignkeydict, columndict)
    foreignkeyclassname = ""
    if documentobject.__class__.__name__ in foreignkeydict:
        if len(foreignkeydict[documentobject.__class__.__name__]) == 0:
            pass #No foreign keys to worry about
        elif len(foreignkeydict[documentobject.__class__.__name__]) == 1:
            (foreignkeyclassname, a) = foreignkeydict[documentobject.__class__.__name__][0]
        else:
            assert False #Only one foreign key allowed
    <fix/>sql = "INSERT INTO " + documentobject.__class__.__name__ + " VALUES (?"
    values = list()</fix>
    for (columnname, columntype) in columndict[documentobject.__class__.__name__]:
        <fix/><fix/>sql += ",?"
</fix></fix>        
        if foreignkeyclassname != "" and foreignkeyclassname + "id" == columnname:
            <fix/><fix/>values.append(parentobject.id)</fix></fix>
        else:
            <fix/><fix/>values.append(getattr(documentobject, columnname))</fix></fix>
    sql += ")"
    <fix/><fix/>self.database.execute(sql, tuple(values))</fix></fix>

firstsaved = False
firstsavededgeid = ""

def SaveEdges(dc, filenameedges, edges):
    c = sqlite3.connect(filenameedges)
    # Create table
    for edge in edges:
        startnodes = list(edge.startnodes)
        if len(edge.startnodes) == 1:
            startnode1id = startnodes[0]
            startnode2id = ""
        elif len(edge.startnodes) == 2:
            startnode1id = startnodes[0]
            startnode2id = startnodes[1]
        else:
            assert False
        
        if edge.propertytype is None:
            propertytypename = ""
        else:
            propertytypename = edge.propertytype
        #try:
        if startnode1id == "":
            global firstsavededgeid
            if firstsavededgeid == "":
                firstsavededgeid = edge.edgeid
            global firstsaved
            assert firstsaved == False or firstsavededgeid == edge.edgeid
            firstsaved = True
        <fix/>c.execute("INSERT INTO edge VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (edge.documentid , edge.documentclassname, 
            edge.__class__.__name__, edge.edgeid, startnode1id, startnode2id, edge.endnode, edge.propertyownerid, edge.propertyname,
            edge.propertyvalue, propertytypename))
        #c.execute("INSERT OR IGNORE INTO edge VALUES ('" + edge.documentid + "', '" + edge.documentclassname + "', '" + edge.__class__.__name__ + "', '" + edge.edgeid + "', " +
        #        "'" + startnode1id + "', '" + startnode2id + "', '" + edge.endnode + "', '" + edge.propertyownerid + "', '" + edge.propertyname + "', '" + str(edge.propertyvalue) + "', "
        #        "'" + propertytypename + "')")</fix>
    c.commit()
    c.close()

    #try:
    #    os.remove(filenamedata)
    #except:
    #    pass
    #database = sqlite3.connect(filenamedata)
    #foreignkeydict = defaultdict(list)
    #for classname in dc.classes:
    #    theclass = dc.classes[classname]
    #    variables = [a for a in dir(theclass) if not a.startswith('__') and not callable(getattr(theclass,a))]
    #    for a in variables:
    #        if isinstance(getattr(theclass, a), FieldList):
    #            foreignkeydict[getattr(theclass, a).theclass.__name__].append((classname, a))
    #columndict = defaultdict(list)
    #for classname in dc.classes:
    #    theclass = dc.classes[classname]
    #    variables = [a for a in dir(theclass) if not a.startswith('__') and not callable(getattr(theclass,a))]
    #    for a in variables:
    #        if isinstance(getattr(theclass, a), FieldList) == False:
    #            columndict[classname].append((a, "int" if isinstance(getattr(theclass, a), FieldInt) else "text"))
    #for k in foreignkeydict:
    #    for (classname, a) in foreignkeydict[k]:
    #        columndict[k].append((classname + "id", "text"))
    #for classname in columndict:
    #    columnlist = columndict[classname]
    #    sql = "CREATE TABLE " + classname + " (id text "
    #    for (a, thetype) in columnlist:
    #        sql += ","
    #        sql += a + " " + thetype
    #    sql += ")"#
    #
    #    database.execute(sql)
    #
    #for documentid in dc.objects:
    #    SaveDocumentObject(database, dc.objects[documentid][0], None, foreignkeydict, columndict)
    #
    #database.commit()

def SaveDocumentObject(database, documentobject, parentobject, foreignkeydict, columndict):
    variables = [a for a in dir(documentobject.__class__) if not a.startswith('__') and not callable(getattr(documentobject.__class__,a))]
    for a in variables:
        if isinstance(getattr(documentobject.__class__, a), FieldList):
            for childobj in getattr(documentobject, a):
                SaveDocumentObject(database, childobj, documentobject, foreignkeydict, columndict)
    foreignkeyclassname = ""
    if documentobject.__class__.__name__ in foreignkeydict:
        if len(foreignkeydict[documentobject.__class__.__name__]) == 0:
            pass #No foreign keys to worry about
        elif len(foreignkeydict[documentobject.__class__.__name__]) == 1:
            (foreignkeyclassname, a) = foreignkeydict[documentobject.__class__.__name__][0]
        else:
            assert False #Only one foreign key allowed
    sql = "INSERT INTO " + documentobject.__class__.__name__ + " VALUES ('" + documentobject.id + "'"
    sql = "INSERT INTO " + documentobject.__class__.__name__ + " VALUES (?"
    values = list()
    for (columnname, columntype) in columndict[documentobject.__class__.__name__]:
        sql += ",?"
        
        if foreignkeyclassname != "" and foreignkeyclassname + "id" == columnname:
            values.append(parentobject.id)
        else:
            values.append(getattr(documentobject, columnname))
    sql += ")"
    self.database.execute(sql, tuple(values))
    
def GetSQLObjects(self, query):
    ret = list()
    cur = self.database.cursor()    
    cur.execute(query)

    rows = cur.fetchall()
    for row in rows:
        for classname in self.documentsbyclass:
            for obj in self.documentsbyclass[classname]:
                if obj.id == row[0]:
                    ret.append(obj)
    return ret
        
def LoadDocumentCollection(dc, filenameedges, filenamedata):
    dc.objects = defaultdict(list)
    #dc.classes = dict()
    dc.historyedgeclasses = dict()
    for theclass in HistoryEdge.__subclasses__():
        dc.historyedgeclasses[theclass.__name__] = theclass

    c = sqlite3.connect(filenameedges)
    cur = c.cursor()    
    c.execute('''CREATE TABLE IF NOT EXISTS edge (
                    documentid text, 
                    documentclassname text, 
                    edgeclassname text, 
                    edgeid text PRIMARY KEY, 
                    startnode1id text, 
                    startnode2id text, 
                    endnodeid text, 
                    propertyownerid text, 
                    propertyname text, 
                    propertyvalue text, 
                    propertytype text
                )''')
    c.commit()
    cur.execute("SELECT documentid, documentclassname, edgeclassname, edgeid, startnode1id, startnode2id, endnodeid, propertyownerid, propertyname, propertyvalue, propertytype FROM edge")

    historygraphdict = defaultdict(HistoryGraph)
    documentclassnamedict = dict()

    rows = cur.fetchall()
    for row in rows:
        documentid = row[0]
        documentclassname = row[1]
        edgeclassname = row[2]
        edgeid = row[3]
        startnode1id = row[4]
        startnode2id = row[5]
        endnodeid = row[6]
        propertyownerid = row[7]
        propertyname = row[8]
        propertyvaluestr = row[9]
        propertytypestr = row[10]

        if documentid in historygraphdict:
            historygraph = historygraphdict[documentid]
        else:
            historygraph = HistoryGraph()
            historygraphdict[documentid] = historygraph
            documentclassnamedict[documentid] = documentclassname
        if propertytypestr == "FieldInt":
            propertyvalue = int(propertyvaluestr)
        elif propertytypestr == "FieldText":
            propertyvalue = str(propertyvaluestr)
        elif propertytypestr == "" and edgeclassname == "HistoryEdgeNull":
            propertyvalue = ""
        else:
            propertyvalue = propertyvaluestr
        documentclassnamedict[documentid] = documentclassname
        if startnode2id == "":
            startnodes = {startnode1id}
        else:
            startnodes = {startnode1id, startnode2id}
        edge = dc.historyedgeclasses[edgeclassname](edgeid, startnodes, endnodeid, propertyownerid, propertyname, propertyvalue, propertytypestr, documentid, documentclassname)
        history = historygraphdict[documentid]
        history.AddEdge(edge)

    nulledges = list()
    for documentid in historygraphdict:
        doc = dc.classes[documentclassnamedict[documentid]](documentid)
        nulledges.extend(history.MergeDanglingBranches())
        history.Replay(doc)
        dc.AddDocumentObject(doc)

    SaveEdges(dc, filenameedges, nulledges)

    return sqlite3.connect(filenamedata) #Return the database that can used for get sql objects

def GetSQLObjects(database, documentcollection, query):
    ret = list()
    cur = database.cursor()    
    cur.execute(query)

    rows = cur.fetchall()
    for row in rows:
        for classname in documentcollection.objects:
            for obj in documentcollection.objects[classname]:
                if obj.id == row[0]:
                    ret.append(obj)
    return ret
        

