import datetime
import json
import logging
import re

import MySQLdb

from collections import namedtuple, defaultdict

logger = logging.getLogger(u'JSON2SQLGenerator')


class JSON2SQLGenerator(object):
    """
    To Generate SQL query from JSON data
    """

    # Constants to map JSON keys
    WHERE_CONDITION = 'where'
    AND_CONDITION = 'and'
    OR_CONDITION = 'or'
    NOT_CONDITION = 'not'
    EXISTS_CONDITION = 'exists'
    CUSTOM_METHOD_CONDITION = 'custom_method'

    # Supported data types by plugin
    INTEGER = 'integer'
    STRING = 'string'
    DATE = 'date'
    DATE_TIME = 'datetime'
    BOOLEAN = 'boolean'
    NULLBOOLEAN = 'nullboolean'
    CHOICE = 'choice'
    MULTICHOICE = 'multichoice'

    CONVERSION_REQUIRED = [
        STRING, DATE, DATE_TIME
    ]

    # Maintain a set of binary operators
    BETWEEN = 'between'
    BINARY_OPERATORS = (BETWEEN, )

    # MySQL aggregate functions
    ALLOWED_AGGREGATE_FUNCTIONS = {'MIN', 'MAX'}

    # Custom methods field types
    ALLOWED_CUSTOM_METHOD_PARAM_TYPES = {'field', 'integer', 'string'}

    # Is operator values
    IS_OPERATOR_VALUE = {'NULL', 'NOT NULL', 'TRUE', 'FALSE'}

    # Supported operators
    VALUE_OPERATORS = namedtuple('VALUE_OPRATORS', [
        'equals', 'greater_than', 'less_than',
        'greater_than_equals', 'less_than_equals',
        'not_equals', 'is_op', 'in_op', 'like', 'between',
        'is_challenge_completed', 'is_challenge_not_completed'
    ])(
        equals='=',
        greater_than='>',
        less_than='<',
        greater_than_equals='>=',
        less_than_equals='<=',
        not_equals='<>',
        is_op='IS',
        in_op='IN',
        like='LIKE',
        is_challenge_completed='is_challenge_completed',
        is_challenge_not_completed='is_challenge_not_completed',
        between=BETWEEN
    )

    DATA_TYPES = namedtuple('DATA_TYPES', [
        'integer', 'string', 'date', 'date_time', 'boolean', 'nullboolean',
        'choice', 'multichoice'
    ])(
        integer=INTEGER,
        string=STRING,
        date=DATE,
        date_time=DATE_TIME,
        boolean=BOOLEAN,
        nullboolean=NULLBOOLEAN,
        choice=CHOICE,
        multichoice=MULTICHOICE
    )

    # Constants
    FIELD_NAME = 'field_name'
    TABLE_NAME = 'table_name'
    DATA_TYPE = 'data_type'

    JOIN_TABLE = 'join_table'
    JOIN_COLUMN = 'join_column'
    PARENT_TABLE = 'parent_table'
    PARENT_COLUMN = 'parent_column'

    CHALLENGE_CHECK_QUERY = 'EXISTS (SELECT 1 FROM journeys_memberstagechallenge WHERE challenge_id = {value} AND ' \
                            'completed_date IS NOT NULL AND member_id = patients_member.id) '

    # - Used in custom method mapping -
    TEMPLATE_STR_KEY = 'template_str'
    TEMPLATE_PARAMS_KEY = 'parameters'
    TEMPLATE_KEY_REGEX = r"{(\w+)}"

    def __init__(self, field_mapping, paths, custom_methods):
        """
        Initialise basic params.
        :type custom_methods: (tuple) tuple of tuples containing (id, sql_template, variables)
        :param field_mapping: (tuple) tuple of tuples containing (field_identifier, field_name, table_name).
        :param paths: (tuple) tuple of tuples containing (join_table, join_field, parent_table, parent_field).
                      Information about paths from a model to reach to a specific model and when to stop.
        :return: None
        """

        self.base_table = ''
        self.field_mapping = self._parse_field_mapping(field_mapping)
        self.path_mapping = self._parse_multi_path_mapping(paths)
        self.custom_methods = self._parse_custom_methods(custom_methods)

        # Mapping to be used to parse various combination keywords data
        self.WHERE_CONDITION_MAPPING = {
            self.WHERE_CONDITION: '_generate_where_phrase',
            self.AND_CONDITION: '_parse_and',
            self.OR_CONDITION: '_parse_or',
            self.NOT_CONDITION: '_parse_not',
            self.EXISTS_CONDITION: '_parse_exists',
            self.CUSTOM_METHOD_CONDITION: '_parse_custom_method_condition',
        }

    def _parse_custom_methods(self, sql_templates):
        """
        Validate the template data and pre process the data.

        :param sql_templates: (tuple) tuple of tuples containing (id, sql_template, variables)
        :return:
        """
        template_mapping = {}

        for template_id, template_str, parameters in sql_templates:
            template_id = int(template_id)
            parameters = json.loads(parameters)
            template_str = template_str.strip()

            assert len(template_str) > 0, 'Not a valid template string'
            assert template_id not in template_mapping, 'Template id must be unique'
            template_defined_variables = set(re.findall(self.TEMPLATE_KEY_REGEX, template_str, re.MULTILINE))
            # Checks if variable defined in template string and variables declared are exactly same
            assert len(set(parameters.keys()) ^ template_defined_variables) == 0, 'Extra variable defined'
            # Checks parameter types are permitted
            assert len(set(map(lambda l: l['data_type'], parameters.values()))
                       - self.ALLOWED_CUSTOM_METHOD_PARAM_TYPES) == 0, 'Invalid data type defined'

            template_mapping[template_id] = {
                self.TEMPLATE_STR_KEY: template_str,
                self.TEMPLATE_PARAMS_KEY: parameters
            }

        return template_mapping

    def _parse_custom_method_condition(self, data):
        """
        Process the custom method condition to render SQL template using the arguments given.

        :param data:
        :return:
        """
        assert isinstance(data, dict), 'Input data must be a dict'
        assert 'template_id' in data, 'No template_id is provided'
        template_id = int(data['template_id'])
        template_data = self.custom_methods[template_id]

        # Process parameters
        validated_parameters = {}
        for param_id, param_data in data.get('parameters', {}).items():
            assert param_id in template_data[self.TEMPLATE_PARAMS_KEY], 'Invalid parameter name.'
            param_type = template_data[self.TEMPLATE_PARAMS_KEY][param_id]['data_type']

            validated_parameters[param_id] = self._process_parameter(param_type, param_data)

        # Check that we have collected all the required keys
        template_params = template_data[self.TEMPLATE_PARAMS_KEY].keys()
        assert len(set(template_params) ^ set(validated_parameters.keys())) == 0, \
            'Missing or extra template variable'

        return template_data[self.TEMPLATE_STR_KEY].format(**validated_parameters)

    def _process_parameter(self, data_type, parameter_data):
        assert len(data_type) > 0, 'Invalid data type'
        assert isinstance(parameter_data, dict), 'Invalid parameter data format'

        if data_type.upper() == 'FIELD':
            field_data = self.field_mapping[parameter_data['field']]
            return "`{table}`.`{field}`".format(
                table=field_data[self.TABLE_NAME], field=field_data[self.FIELD_NAME]
            )
        elif data_type.upper() == 'INTEGER':
            return int(parameter_data['value'])
        elif data_type.upper() == 'STRING':
            <fix/>return "'{}'".format(self._sql_injection_proof(parameter_data['value']))</fix>
        else:
            raise AttributeError("Unsupported data type for parameter: {}".format(data_type))

    def generate_sql(self, data, base_table):
        """
        Create SQL query from provided json
        :param data: (dict) Actual JSON containing nested condition data.
                     Must contain two keys - fields(contains list of fields involved in SQL) and where_data(JSON data)
        :param base_table: (string) Exact table name as in DB to be used with FROM clause in SQL.
        :return: (unicode) Finalized SQL query unicode
        """
        self.base_table = base_table
        assert self.validate_where_data(data.get('where_data', {})), 'Invalid where data'
        where_phrase = self._generate_sql_condition(data['where_data'])

        if 'group_by_fields' in data:
            assert isinstance(data['group_by_fields'], list), 'Group by fields need to list of dict'
            data['group_by_fields'] = list(map(lambda x: int(x['field']), data['group_by_fields']))

        path_subset = self.extract_paths_subset(list(
            map(lambda field_id: self.field_mapping[field_id][self.TABLE_NAME], data['fields'])),
            data.get('path_hints', {})
        )
        join_tables = self.create_join_path(path_subset, self.base_table)
        join_phrase = self.generate_left_join(join_tables)
        group_by_phrase = self.generate_group_by(data.get('group_by_fields', []),
                                                 data.get('having', {}))
        count_phrase = u'COUNT(DISTINCT `{base_table}`.`id`)'.format(base_table=base_table)

        return u'SELECT {count_phrase} FROM {base_table} {join_phrase} WHERE {where_phrase} {group_by_fragment}'.format(
            join_phrase=join_phrase,
            base_table=base_table,
            where_phrase=where_phrase,
            group_by_fragment=group_by_phrase,
            count_phrase=count_phrase
        )

    def _parse_multi_path_mapping(self, paths):
        """
        Create mapping of what nodes can be reached from any given node.
        This method also support the case when you can jump to multiple node from any given node

        :param paths: (tuple) tuple of tuples in the format ((join_table, join_field, parent_table, parent_field),)
        :return: (dict) dict in the format {'join_table': {'parent_table': {'parent_field': , 'join_field':} }}
        """
        path_map = defaultdict(dict)
        for join_tbl, join_fld, parent_tbl, parent_fld in paths:
            # We can support if there are multiple ways to join a table
            # We don't support if there are multiple fields on join table path
            assert parent_tbl not in path_map[join_tbl], 'Joins with multiple fields is not supported'
            path_map[join_tbl][parent_tbl] = {
                self.PARENT_COLUMN: parent_fld,
                self.JOIN_COLUMN: join_fld
            }

        return path_map

    def extract_paths_subset(self, start_nodes, path_hints):
        """
        Extract a subset of paths which only contains paths which are possible from starting nodes
        When there is multiple options from any node then we look in path hints to select a node.

        This method also aims to merge duplicate path nodes.
        Example:
        A -> B -> C
        A -> B ->  D

        Merge this into
        A -> B -> C
             | -> D

        Merge happens from left side not right side.
        As our current implementation base is always on left side

        Left side is always base_table
        :param start_nodes: Array of table names
        :param path_hints:
        :return:
        """
        path_subset = defaultdict(set)
        # Convert start nodes to set as we would need this for lookups
        start_nodes = set(start_nodes)
        traversal_nodes = list(start_nodes)  # type: list

        # We would be doing traversal from given tables towards base tables.
        while len(traversal_nodes) > 0:
            curr_node = traversal_nodes.pop()  # type: str

            # This condition indicate that we have reached end of path
            if curr_node == self.base_table:
                continue

            next_nodes = self.path_mapping[curr_node]  # type: dict

            if curr_node in path_hints:
                assert path_hints[curr_node] in next_nodes, 'Node provided in hint is not a valid option.'
                assert len(
                    set(self.path_mapping[curr_node]) &
                    (start_nodes | set(path_hints.values()))
                ) == 1, 'Multiple paths are selected from node {}'.format(curr_node)
                parent_node = path_hints[curr_node]
                traversal_nodes.append(parent_node)
            elif len(next_nodes) == 1:
                parent_node = next_nodes.keys()[0]
                traversal_nodes.append(parent_node)
            else:
                raise Exception("No path hint provided for `{}`".format(curr_node))

            path_subset[parent_node].add(curr_node)

        return path_subset

    def create_join_path(self, path_map, curr_table):
        """
        Convert the path subset into a join table
        Return list of tuples
        [(join table, parent table)]
        :param path_map: (dict) Nested dict of format { join_table: { parent_table: {join_field, parent_field} } }
        :param curr_table: (str) Node from which we need to be created.
        :return:
        """
        # This condition satisfied when we reach the end of the current path
        if curr_table not in path_map:
            return

        for table_name in sorted(path_map[curr_table]):
            yield (table_name, curr_table)
            for item in (self.create_join_path(path_map, table_name)):
                yield item

    def generate_left_join(self, join_path):
        join_phrases = []
        for join_table, parent_table in join_path:
            join_phrases.append(
                'LEFT JOIN {join_tbl} ON {join_tbl}.{join_fld} = {parent_tbl}.{parent_fld}'.format(
                    join_tbl=join_table,
                    parent_tbl=parent_table,
                    join_fld=self.path_mapping[join_table][parent_table][self.JOIN_COLUMN],
                    parent_fld=self.path_mapping[join_table][parent_table][self.PARENT_COLUMN]
                )
            )

        return ' '.join(join_phrases)

    def generate_group_by(self, group_by_fields, having_clause):
        """
        Validate and return group by and having clause statement

        :rtype: str
        :type having_clause: Dict
        :type group_by_fields: List[int]
        """
        assert isinstance(group_by_fields, list)
        assert isinstance(having_clause, dict)

        assert self.validate_group_by_data(group_by_fields, having_clause), 'Invalid having data'

        if len(group_by_fields) == 0:
            return ''

        result = ''
        fully_qualified_field_names = map(
            lambda field_id: '`{table_name}`.`{field_name}`'.format(
                table_name=self.field_mapping[field_id][self.TABLE_NAME],
                field_name=self.field_mapping[field_id][self.FIELD_NAME]
            ),
            group_by_fields
        )

        result += 'GROUP BY {fields}'.format(fields=', '.join(fully_qualified_field_names))
        if len(having_clause.keys()) > 0:
            result += ' HAVING {condition}'.format(condition=self._generate_sql_condition(having_clause))

        return result

    def validate_group_by_data(self, group_by_fields, having):
        """
        Validate the group by data to check if it can produce a query which is valid.
        For example it would check only group by fields or aggregate functions are being used.

        :type having: Dict
        :type group_by_fields: List[int]
        """
        assert isinstance(group_by_fields, list)
        assert isinstance(having, dict)

        for cond in self.extract_key_from_nested_dict(having, self.WHERE_CONDITION):
            assert isinstance(cond, dict), 'where condition needs to be dict'
            assert 'aggregate_lhs' in cond or cond.get('field') in group_by_fields, \
                'Use of non aggregate value or non grouped field: {}'.format(cond)

        return True

    def validate_where_data(self, where_data):
        """
        Validate if where fields doesn't contains use of aggregation function
        :type where_data: Dict
        """
        assert isinstance(where_data, dict) and len(where_data) > 0, \
            'Invalid or empty where data'

        for cond in self.extract_key_from_nested_dict(where_data, self.WHERE_CONDITION):
            assert isinstance(cond, dict), 'Invalid where condition'
            assert cond.get('aggregate_lhs', '') == '', \
                'Use of non aggregate value or non grouped field: {}'.format(cond)

        return True

    def _generate_sql_condition(self, data):
        """
        This function uses recursion to generate sql for nested conditions.
        Every key in the dict will map to a function by referencing WHERE_CONDITION_MAPPING.
        The function mapped to that key will be responsible for generating SQL for that part of the data.
        :param data: (dict) Conditions data which needs to be parsed to generate SQL
        :return: (unicode) Unicode representation of data into SQL
        """
        result = ''
        # Check if data is not blank
        if data:
            # Get the first key in dict.
            condition = data.keys()[0]
            # Call the function mapped to the condition
            function = getattr(self, self.WHERE_CONDITION_MAPPING.get(condition))
            result = function(data[condition])
        return result

    def _get_validated_data(self, where):
        try:
            operator = where['operator'].lower()
            value = where['value']
            field = where['field']
        except KeyError as e:
            raise KeyError(
                u'Missing key - [{}] in where condition dict'.format(e.args[0])
            )
        else:
            # Get optional secondary value
            secondary_value = where.get('secondary_value')
            # Check if secondary_value is present for binary operators
            if operator in self.BINARY_OPERATORS and not secondary_value:
                raise ValueError(
                    u'Missing key - [secondary_value] for operator - [{}]'.format(
                        operator
                    )
                )
            return operator, value, field, secondary_value

    def _generate_where_phrase(self, where):
        """
        Function to generate a single condition(column1 = 1, or column1 BETWEEN 1 and 5) based on data provided.
        Uses _join_names to assign table_name to a field in query.
        :param where: (dict) will contain required data to generate condition.
                      Sample Format: {"field": , "primary_value": ,"operator": , "secondary_value"(optional): }
        :return: (unicode) SQL condition in unicode represented by where data
        """
        # Check data valid
        if not isinstance(where, dict):
            raise ValueError(
                'Where condition data must be a dict. Found [{}]'.format(
                    type(where)
                )
            )
        # Get all the data elements required and validate them
        operator, value, field, secondary_value = self._get_validated_data(where)
        # Get db field name
        field_name = self.field_mapping[field][self.FIELD_NAME]
        # Get corresponding SQL operator
        sql_operator = getattr(self.VALUE_OPERATORS, operator)
        # Get data type and table name from field_mapping
        data_type = self._get_data_type(field)
        table = self._get_table_name(field)

        # `value` contains the R.H.S part of the equation.
        # In case of `IS` operator R.H.S can be `NULL` or `NOT NULL`
        # irrespective of data type of the L.H.S.
        # Hence we want to skip data type check for `IS` operator.
        if sql_operator == self.VALUE_OPERATORS.is_op:
            assert value.upper() in self.IS_OPERATOR_VALUE, 'Invalid rhs for `IS` operator'
            sql_value, secondary_sql_value = value.upper(), None
        else:
            # Check if the primary value and data_type are in sync
            self._sanitize_value(value, data_type)
            # Check if the secondary_value and data_type are in sync
            if secondary_value:
                self._sanitize_value(secondary_value, data_type)

            # Make string SQL injection proof
            if data_type == self.STRING:
                value = self._sql_injection_proof(value)
                if secondary_value:
                    secondary_value = self._sql_injection_proof(secondary_value)

            # Make value sql proof. For ex: if value is string or data convert it to '<value>'
            sql_value, secondary_sql_value = self._convert_values(
                [value, secondary_value], data_type
            )

        lhs = u'`{table}`.`{field}`'.format(table=table, field=field_name)  # type: unicode

        # Apply aggregate function to L.H.S
        if 'aggregate_lhs' in where:
            aggregate_func_name = where['aggregate_lhs'].upper()  # type: unicode
            if aggregate_func_name in self.ALLOWED_AGGREGATE_FUNCTIONS:
                lhs = u'{func_name}({field_name})'.format(func_name=aggregate_func_name, field_name=lhs)
            else:
                logger.info("Unsupported aggregate functions: {}".format(aggregate_func_name))

        # TODO: Based on the assumption that below operator will only used
        #           with challenge.
        if sql_operator in [self.VALUE_OPERATORS.is_challenge_completed,
                            self.VALUE_OPERATORS.is_challenge_not_completed]:
            return "{negate} {check}".format(
                negate='NOT' if sql_operator == self.VALUE_OPERATORS.is_challenge_not_completed else '',
                check=self.CHALLENGE_CHECK_QUERY.format(value=sql_value)
            )

        # Generate SQL phrase
        if sql_operator == self.BETWEEN:
            where_phrase = u'{lhs} {operator} {primary_value} AND {secondary_value}'.format(
                lhs=lhs, operator=sql_operator,
                value=sql_value, secondary_value=secondary_sql_value
            )
        else:
            where_phrase = u'{lhs} {operator} {value}'.format(
                operator=sql_operator, lhs=lhs, value=sql_value,
            )
        return where_phrase

    def _get_data_type(self, field):
        """
        Gets data type for the field from self.field_mapping configured in __init__
        :param field: (int|string) field identifier that is used as key in self.field_mapping
        :return: (string) data type of the field
        """
        return self.field_mapping[field][self.DATA_TYPE]

    def _get_table_name(self, field):
        """
        Gets table name for the field from self.field_mapping configured in __init__
        :param field: (int|string) Field identifier that is used as key in self.field_mapping
        :return: (string|unicode) Name of the table of the field
        """
        return self.field_mapping[field][self.TABLE_NAME]

    def _convert_values(self, values, data_type):
        """
        Converts values for SQL query. Adds '' string, date, datetime values, string choice value
        :param values: (iterable) Any instance of iterable values of same data type that need conversion
        :param data_type: (string) Data type of the values provided
        """
        if data_type in [self.CHOICE, self.MULTICHOICE]:
            # try converting the value to int
            try:
                int(values[0])
            except ValueError:
                wrapper = '\'{value}\''
            else:
                wrapper = '{value}'
        elif data_type in self.CONVERSION_REQUIRED:
            wrapper = '\'{value}\''
        else:
            wrapper = '{value}'
        return (wrapper.format(value=value) for value in values)

    def _sanitize_value(self, value, data_type):
        """
        Validate value with data type
        :param value: Values that needs to be validated with data type
        :param data_type: (string) Data type against which the value will be compared
        :return: None
        """
        if data_type == self.INTEGER:
            try:
                int(value)
            except ValueError:
                raise ValueError(
                    'Invalid value -[{}] for data_type - [{}]'.format(
                        value, data_type
                    )
                )
        elif data_type == self.DATE:
            try:
                datetime.datetime.strptime(value, '%Y-%m-%d')
            except ValueError as e:
                raise e
        elif data_type == self.DATE_TIME:
            try:
                datetime.datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')
            except ValueError as e:
                raise e

    def _parse_and(self, data):
        """
        To parse the AND condition for where clause.
        :param data: (list) contains list of data for conditions that need to be ANDed
        :return: (unicode) unicode containing SQL condition represeted by data ANDed. 
                 This SQL can be directly placed in a SQL query
        """
        return self._parse_conditions(self.AND_CONDITION, data)

    def _parse_or(self, data):
        """
        To parse the OR condition for where clause.
        :param data: (list) contains list of data for conditions that need to be ORed
        :return: (unicode) unicode containing SQL condition represeted by data ORed. 
                 This SQL can be directly placed in a SQL query
        """
        return self._parse_conditions(self.OR_CONDITION, data)

    def _parse_exists(self, data):
        """
        To parse the EXISTS check/wrapper for where clause.
        :param data: (list) contains a list of single element of data for conditions that
                            need to be wrapped with a EXISTS check in WHERE clause
        :return: (unicode) unicode containing SQL condition represeted by data with EXISTS check. 
                 This SQL can be directly placed in a SQL query
        """
        raise NotImplementedError

    def _parse_not(self, data):
        """
        To parse the NOT check/wrapper for where clause.
        :param data: (list) contains a list of single element of data for conditions that
                            need to be wrapped with a NOT check in WHERE clause
        :return: (unicode) unicode containing SQL condition represeted by data with NOT check. 
                 This SQL can be directly placed in a SQL query
        """
        return self._parse_conditions(self.NOT_CONDITION, data)

    def _parse_conditions(self, condition, data):
        """
        To parse AND, NOT, OR, EXISTS data and
        delegate to proper functions to generate combinations according to condition provided.
        NOTE: This function doesn't do actual parsing. 
              All it does is deligate to a function that would parse the data.
              The main logic for parsing only resides in _generate_where_phrase
              as every condition is similar, its just how we group them
        :param condition: (string) the condition to use to combine the condition represented by data
        :param data: (list) list conditions to be combined or parsed
        :return: (unicode) unicode string that could be placed in the SQL
        """
        sql = bytearray()
        for element in data:
            # Get the first key in the dict.
            inner_condition = element.keys()[0]
            function = getattr(self, self.WHERE_CONDITION_MAPPING.get(inner_condition))
            # Call the function mapped to it.
            result = function(element.get(inner_condition))
            # Append the result to the sql.
            if not sql and condition in [self.AND_CONDITION, self.OR_CONDITION]:
                sql.extend('({})'.format(result))
            else:
                sql.extend(' {0} ({1})'.format(condition, result))
        return u'({})'.format(sql.decode('utf8'))

    def _parse_field_mapping(self, field_mapping):
        """
        Converts tuple of tuples to dict.
        :param field_mapping: (tuple) tuple of tuples in the format ((field_identifier, field_name, table_name, data_type),)
        :return: (dict) dict in the format {'<field_identifier>': {'field_name': <>, 'table_name': <>, 'data_type': <>,}}
        """
        return {
            field[0]: {
               self.FIELD_NAME: field[1],
               self.TABLE_NAME: field[2],
               self.DATA_TYPE: field[3]
            } for field in field_mapping
        }

    def _sql_injection_proof(self, value):
        """
        Escapes strings to avoid SQL injection attacks
        :param value: (string|unicode) string that needs to be escaped
        :return: (string|unicode) escaped string
        """
        return MySQLdb.escape_string(value)

    def extract_key_from_nested_dict(self, target_dict, key):
        """
        Traverse the dictionary recursively and return the value with specified key

        :type target_dict: Dict
        :type key: str
        """
        assert isinstance(target_dict, dict)
        assert isinstance(key, str) and key

        for k, v in target_dict.items():
            if k == key:
                yield v
            elif isinstance(v, dict):
                for item in self.extract_key_from_nested_dict(v):
                    yield item
