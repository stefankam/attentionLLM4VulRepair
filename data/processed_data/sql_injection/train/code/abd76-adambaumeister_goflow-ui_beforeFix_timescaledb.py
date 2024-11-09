from gfui.backends.default import Backend
import psycopg2
from gfui.chartgraph import Graph, Table
import re
import ipaddress
import os

class Timescaledb_backend(Backend):
    def __init__(self, OPTIONS):
        super().__init__()
        self.required_opts = ['SQL_SERVER', 'SQL_USERNAME', 'SQL_DB']
        self.parse_options(OPTIONS)
        self.columns = {}

        pw = os.environ.get("SQL_PASSWORD")
        if not pw:
            pw = self.OPTIONS['SQL_PASSWORD']

        self.db = psycopg2.connect(
            "dbname={0} user={1} password={2} host={3}".format(
                self.OPTIONS['SQL_DB'],
                self.OPTIONS['SQL_USERNAME'],
                pw,
                self.OPTIONS['SQL_SERVER']
            )
        )

        self.schema = Schema()

        self.filters = []

    def get_columns(self):
        return self.schema.get_columns()

    def add_filter(self, op, value):
        self.schema.add_filter(value, op)

    def get_int_columns(self):
        return self.schema.get_int_columns()

    def flow_table(self, limit=10):
        db = self.db
        self.schema.limit = limit
        FLOWS = self.schema.flows()

        cursor = self.schema.query(db, FLOWS)
        r = cursor.fetchall()
        t = Table()
        t = t.table_from_rows(r, self.schema.column_order)
        return t

    def topn_sum_graph(self, field, sum_by, limit=10):
        db = self.db
        self.schema.limit = limit
        FLOWS_PER_IP = self.schema.topn_sum(field, sum_by)

        <vul/>cursor = db.cursor()
        cursor.execute(FLOWS_PER_IP)</vul>
        r = cursor.fetchall()
        g = Graph()
        g.name = "TopN {0}".format(field)
        g.set_headers([
            field,
            "Total"
        ])
        g.graph_from_rows(r, 0)
        return g

class Column:
    """
    Column

    Column handling class.
    Governs how query strings are built and helper functons for returned data.
    """
    def __init__(self, name, display_name=None):
        self.name = name
        self.display_name = display_name
        self.type = 'text'
        self.filter_string = None

    def get_display_name(self):
        return self.display_name

    def select(self):
        return "{0}".format(self.name)

    def filter(self, value, op=None):
        if self.filter_string:
            self.filter_string = self.filter_string + "AND {2} {0} \"{1}\"".format(op, value, self.name)
        else:
            self.filter_string = "{2} {0} \"{1}\"".format(op, value, self.name)

class IP4Column(Column):
    def __init__(self, name, display_name=None):
        super().__init__(name, display_name)
        self.type = "ip"

    def select(self):
        return "{0}".format(self.name)

    def filter(self, value, op=None):
        s = value.split("/")
        if len(s) > 1:
            <vul/>self.filter_string = "({0} << '{1}'".format(self.name, value)</vul>
        else:
            <vul/>self.filter_string = "{0} = '{1}'".format(self.name, value)</vul>

        return self.filter_string

class IP6Column(Column):
    def __init__(self, name, display_name=None):
        super().__init__(name, display_name)
        self.type = "ip6"

    def select(self):
        return "{0}".format(self.name)

    def filter(self, value, op=None):
        s = value.split("/")
        if len(s) > 1:
            <vul/>ip = ipaddress.ip_network(value, strict=False)
            start_ip = ip.network_address
            end_ip = ip.broadcast_address
            self.filter_string = "({0} > {1} AND {0} < {2})".format(self.name, int(start_ip), int(end_ip))</vul>
        else:
            <vul/>ip = ipaddress.ip_address(value)
            self.filter_string = "{0} = {1}".format(self.name, int(ip))</vul>

        return self.filter_string

class IntColumn(Column):
    def __init__(self, name, display_name=None):
        super().__init__(name, display_name)
        self.type = 'int'

    def select(self):
        return "{0}".format(self.name)

    def filter(self, value, op=None):
        self.filter_string = "{0} = {1}".format(self.name, value)
        return self.filter_string

class PortColumn(Column):
    def __init__(self, name, display_name=None):
        super().__init__(name, display_name)
        self.type = 'port'

    def select(self):
        return "{0}".format(self.name)

    def filter(self, value, op=None):
        self.filter_string = "{0} = %s".format(self.name, value)
        return self.filter_string

class Coalesce:
    def __init__(self, name, columns, filter_func, display_name):
        """
        Coalesce
        Select from a list of columns whatever is not null
        :param columns (List): Column objects
        """
        self.name = name
        self.columns = columns
        # We assume that the passed columns are of roughly the same type
        self.type = columns[0].type
        self.column_selects = []
        for c in columns:
            self.column_selects.append(c.select())

        self.filter_string = None
        self.filter_func = filter_func
        self.display_name = display_name

    def get_display_name(self):
        return self.display_name

    def select(self):
        fields = ", ".join(self.column_selects)
        return "COALESCE({0}) AS {1}".format(fields, self.name)

    def filter(self, value, op=None):
        self.filter_string = self.filter_func(value, op)

class Schema:
    """
    Schema

    Defines the backend schema
    Changes to the backend (naming, etc.) should be reflected here.
    """
    def __init__(self):
        # Default
        self.limit = 10

        self.column_order = [
            "last_switched",
            "src_ip",
            "src_port",
            "dst_ip",
            "dst_port",
            "in_bytes",
        ]
        src_ip_col = IP4Column("src_ip", "Source IP")
        src_ipv6_col = IP6Column("src_ipv6", "Source IPv6")
        dst_ip_col = IP4Column("dst_ip", "Destination IP")
        dst_ipv6_col = IP6Column("dst_ipv6", "DestinationIPv6")

        self.filter_val_list = []

        # Columns
        self.columns = {
            "last_switched": Column("last_switched", "Last Switched"),
            "src_ip": Coalesce("src_c_ip", [src_ip_col, src_ipv6_col], src_ip_col.filter, "Source IP"),
            "src_port": PortColumn("src_port", "Source Port"),
            "dst_ip": Coalesce("dst_c_ip", [dst_ip_col, dst_ipv6_col], dst_ip_col.filter, "Destination IP"),
            "dst_port": PortColumn("dst_port", "Destination Port"),
            "in_bytes": IntColumn("in_bytes", "Input bytes"),
            "in_pkts": IntColumn("in_pkts", "Input Packets"),
        }

        # Supported queries
        self.QUERIES = {
            "TOPN": self.topn
        }

        self.filters = []

        self.filter_map = {
            "(\d+\-\d+\-\d+)": "last_switched",
            "src (\d+\.\d+\.\d+\.\d+\/\d+|\d+\.\d+\.\d+\.\d+)": "src_ip",
            "dst (\d+\.\d+\.\d+\.\d+\/\d+|\d+\.\d+\.\d+\.\d+)": "dst_ip",
            "src ([0-9]+)($|\s)": "src_port",
            "dst ([0-9]+)($|\s)": "dst_port",
        }

    def add_filter(self, value, op="="):
        for regex, column in self.filter_map.items():
            if re.search(regex, value):
                m = re.search(regex, value)
                v = m.group(1)
                self.columns[column].filter(v, op)
                self.filter_val_list.append(v)

    def build_filter_string(self):
        s = 'WHERE '
        l = []
        for c in self.columns.values():
            if c.filter_string:
                l.append(c.filter_string)

        if len(l) > 0:
            return s + " AND ".join(l)
        else:
            return ''

    def get_columns(self):
        result = {}
        for col_name, col in self.columns.items():
            result[col_name] = col.get_display_name()

        return result

    def get_int_columns(self):
        result = {}
        for col_name, col in self.columns.items():
            if col.type is "int":
                result[col_name] = col.get_display_name()

        return result

    def topn(self, column):
        count = "last_switched"
        q = """
        SELECT {0}, count({1}) AS c FROM goflow_records {2} GROUP BY {0} ORDER BY c DESC
        """.format(self.columns[column].select(), count, self.build_filter_string())
        return self.query_boilerplate(q)

    def topn_sum(self, column, sum_by):
        q = """
        SELECT {0}, sum({1}) AS c FROM goflow_records {2} GROUP BY {3} ORDER BY c DESC
        """.format(self.columns[column].select(), sum_by, self.build_filter_string(), self.columns[column].name)
        return self.query_boilerplate(q)

    def flows(self):
        c = []
        for col in self.column_order:
            c.append(self.columns[col].select())
        q = """
        SELECT {1} FROM goflow_records {0} ORDER BY last_switched DESC
        """.format(self.build_filter_string(), ", ".join(c))
        return self.query_boilerplate(q)

    def query_boilerplate(self, q):
        q = q + """LIMIT {0}""".format(self.limit)
        return q

    def query(self, db, q):
        cursor = db.cursor()
        cursor.execute(q, self.filter_val_list)
        return cursor