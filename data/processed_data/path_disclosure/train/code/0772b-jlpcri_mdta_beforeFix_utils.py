import collections
import time

from mdta.apps.graphs.models import Node, Edge
from mdta.apps.projects.models import Project, TestRailConfiguration
from mdta.apps.testcases.testrail import APIClient, APIError
from mdta.apps.testcases.utils_backwards_traverse import path_traverse_backwards
from mdta.apps.testcases.utils_negative_testcases import negative_testcase_generation, rejected_testcase_generation

from mdta.apps.testcases.constant_names import NODE_START_NAME, NODE_MP_NAME, LANGUAGE_DEFAULT_NAME


def context_testcases():
    """
    Retrieve context for tab TestCases
    :return:
    """
    context = {
        'projects': Project.objects.all(),
        'testrails': TestRailConfiguration.objects.all(),
    }

    return context


# --------------- Routing Project/Module Graph Start ---------------
def create_routing_test_suite(project=None, modules=None):
    """
    Create routing paths for project.modules lists or module lists
    :param project:
    :param modules:
    :return:
    """
    data = []

    if project:
        if project.language:
            language = project.language.name
        else:
            language = LANGUAGE_DEFAULT_NAME
        <vul/>start = time.time()
        data = create_routing_test_suite_module(project.modules, language)
        print(project.name, time.time() - start)</vul>
    elif modules:
        if modules[0].project.language:
            language = modules[0].project.language.name
        else:
            language = LANGUAGE_DEFAULT_NAME
        <vul/>data = create_routing_test_suite_module(modules, language)</vul>

    return data


<vul/>def create_routing_test_suite_module(modules, language):</vul>
    """
    Create routing paths for list of modules
    :param modules:
    :return:
    """
    test_suites = []

    if len(modules) > 0 and modules[0].project.test_header:
        th_module = modules[0].project.test_header
    else:
        th_module = None

    for module in modules:
        <vul/>start_time = time.time()
        data = get_paths_through_all_edges(module.edges_all, th_module, language)</vul>

        test_suites.append({
            'module': module.name,
            'data': data
        })
        <vul/>print(module.name, time.time() - start_time)</vul>

    return test_suites


<vul/>def get_paths_through_all_edges(edges, th_module=None, language=None):</vul>
    """
    Get all paths through all edges
    :param edges:
    :return:
    """
    th_paths = get_paths_from_test_header(th_module)
    # print(th_paths)

    shortest_set = []  # found shortest set from Start to node, key is 'Start + node', value is list of nodes

    data = []
    if th_paths:
        for th_path in th_paths:
            for edge in edges:
                # print(edge.id)
                start = time.time()
                path = routing_path_to_edge(edge, shortest_set)
                # print(edge.from_node.name, '-', edge.to_node.name, time.time() - start)

                if path:
                    path_data = path_traverse_backwards(path, th_path=th_path, language=language)
                    if 'tcs_cannot_route' in path_data.keys():
                        data.append({
                            'tcs_cannot_route': path_data['tcs_cannot_route'],
                            'id': edge.id,
                            'title': 'Route from \'' +
                                     edge.from_node.name +
                                     '\' to \'' +
                                     edge.to_node.name + '\''
                        })
                    else:
                        title = 'Route from \'' + edge.from_node.name +\
                                    '\' to \'' + edge.to_node.name + '\''
                        # edge_id = edge.id,
                        data.append({
                                'pre_conditions': path_data['pre_conditions'],
                                'tc_steps': path_data['tc_steps'],
                                'id': edge.id,
                                'title': title,
                            })

                        if edge.to_node.type.name in NODE_MP_NAME:
                            negative_testcase_generation(data, path_data, title, edge.to_node, edge, language=language)
                            if edge.to_node.type.name == NODE_MP_NAME[1]:
                                rejected_testcase_generation(data, path_data, title, edge.to_node, edge, language=language)

    else:
        for edge in edges:
            path = routing_path_to_edge(edge, shortest_set)

            if path:
                path_data = path_traverse_backwards(path, language=language)
                if 'tcs_cannot_route' in path_data.keys():
                    data.append({
                        'tcs_cannot_route': path_data['tcs_cannot_route'],
                        'id': edge.id,
                        'title': 'Route from \'' +
                                 edge.from_node.name +
                                 '\' to \'' +
                                 edge.to_node.name + '\''
                    })
                    print(edge.id)
                else:
                    title = 'Route from \'' + edge.from_node.name +\
                                    '\' to \'' + edge.to_node.name + '\''
                    # edge_id = edge.id,
                    data.append({
                            'pre_conditions': path_data['pre_conditions'],
                            'tc_steps': path_data['tc_steps'],
                            'id': edge.id,
                            'title': title
                        })

                    if edge.to_node.type.name == NODE_MP_NAME[0]:
                        negative_testcase_generation(data, path_data, title, edge.to_node, edge, language=language)

    # return check_subpath_in_all(data)
    return data


def routing_path_to_edge(edge, shortest_set=None):
    """
    Routing path to current Edge, edge.from_node
    :param edge:
    :return:
    """
    data = routing_path_to_node(edge.from_node, shortest_set)

    if data:
        data.append(edge)
        data.append(edge.to_node)

    # print(data)
    return data


def routing_path_to_node(node, shortest_set=None):
    """
    Routing path to current Node
    :param node:
    :return:
    """
    path = backwards_search(node, shortest_set)

    # print(path)
    return path


def backwards_search(node, shortest_set=None):
    """
    Search a path from Start node(type='Start') to current Node
    Breadth
    :param node:
    :return:
    """

    path = []
    start_node_found_outside = False  # flag to find Start Node outside

    if node.type.name in NODE_START_NAME:
        path.append(node)
    else:
        edges = node.arriving_edges
        if edges.count() == 1:
            edge = edges[0]
        elif edges.count() > 1:
            edge = get_shortest_edge_from_arriving_edges(node, shortest_set)
        else:
            edge = None

        if edge:
            start_node_found = False  # flag to find Start Node in current search
            if edge.from_node.type.name not in NODE_START_NAME:
                if edge.from_node.arriving_edges.count() > 0:
                    start_node_found_outside = True
                    path += backwards_search(edge.from_node, shortest_set)
            else:
                start_node_found = True
                path.append(edge.from_node)

            if start_node_found or start_node_found_outside:  # if found Start Node, add Edge
                path.append(edge)
                path.append(node)

    # print('path: ',  path)
    return path


def check_subpath_in_all(all_path):
    """
    Find sub paths contained by parent path in all possible paths and remove them
    Use set(a) < set(b) to compare if list_b contains list_a
    :param all_path:
    :return:
    """
    data = []
    length = len(all_path)
    for i in range(length):
        for j in range(i + 1, length):
            if set(all_path[i]) < set(all_path[j]) and not check_path_contains_in_result(all_path[j], data):
                data.append(all_path[j])
            elif set(all_path[i]) > set(all_path[j]) and not check_path_contains_in_result(all_path[i], data):
                data.append(all_path[i])

    # check if first path of all_path is in result
    if length > 0 and not check_path_contains_in_result(all_path[0], data):
        data.append(all_path[0])

    return data


def check_path_contains_in_result(path, result):
    """
    Check current path is covered in result
    :param path:
    :param result:
    :return:
    """
    flag = False
    for i in range(len(result)):
        if set(path) <= set(result[i]):
            flag = True
            break
        else:
            continue

    return flag


def get_shortest_edge_from_arriving_edges(node, shortest_set=None):
    if node.module.project:
        start_nodes = node.module.project.start_nodes
    else:
        start_nodes = node.module.start_nodes

    edge = ''
    for start_node in start_nodes:
        tmp = start_node.name + '-' + node.name
        exist_shortest = next((item for item in shortest_set if item['name'] == tmp), None)
        if exist_shortest:
            path = exist_shortest['path']
        else:
            # print(tmp)
            path = breadth_first_search(start_node, node)
            shortest_set.append({
                'name': tmp,
                'path': path
            })
        for each in path[:-1]:
            edges = Edge.objects.filter(from_node=each, to_node=node)
            if edges.count() > 0:
                edge = edges[0]
            # try:
            #     edge = Edge.objects.get(from_node=each, to_node=node)
            #     return edge
            # except Edge.DoesNotExist:
            #     pass

    return edge


def breadth_first_search(start, end):
    visited = [start]
    queue = collections.deque([start])
    while queue:
        vertex = queue.popleft()
        for child in vertex.children:
            if child not in visited:
                if child == end:
                    visited.append(child)
                    break
                visited.append(child)
                queue.append(child)

    return visited


# --------------- Routing Project/Module Graph End ---------------


# --------------- Routing Test Header Graph End ---------------
def get_paths_from_test_header(th_module):
    """
    Search paths from TestHeader and put it between Start Node and next Node
    :param th_module: TestHeader module
    :return: TestHeader route paths
    """
    data = []
    if th_module:
        try:
            end_node = Node.objects.get(module=th_module, type__name='TestHeader End')
            for edge in end_node.arriving_edges:
                path = routing_path_to_edge(edge)
                data.append(path)
        except Exception as e:
            print('TestHeader error: ', e)
            pass

    return data
# --------------- Routing Test Header Graph End ---------------


# --------------- TestRail Start ---------------
def get_projects_from_testrail(instance):
    """
    Get Projects from TestRail to help adding TestRail Configuration
    :param instance: TestRail instance
    :return: Projects of TestRail
    """
    client = APIClient(instance.host)
    client.user = instance.username
    client.password = instance.password

    return client.send_get('get_projects')


def add_testsuite_to_project(client, project_id, suite_name):
    """
    Add TestSuite to Project on TestRail
    :param client:
    :param project_id: Project ID of TestRail
    :param suite_name: TestSuite name of TestRail Project, same as MDTA project version name
    :return:
    """
    data = {
        'name': suite_name,
        'description': ''
    }

    try:
        suite = client.send_post('add_suite/' + project_id, data)
        return suite
    except APIError as e:
        print('Add Suite Error: ', e)
        return None


def add_section_to_testsuite(client, project_id, suite_id, section_name):
    """
    Add section to TestSuite of Testrail project
    :param client:
    :param project_id: Project ID of TestRail
    :param suite_id: TestSuite ID of TestRail Project == MDTA.project.version
    :param section_name: Section name of TestRail-Project-TestSuite == MDTA.project.module.name
    :return:
    """
    data = {
        'suite_id': suite_id,
        'name': section_name
    }

    section = client.send_post('add_section/' + project_id, data)

    return str(section['id'])


def remove_section_from_testsuite(client, section_id):
    """
    Delete section from TestRail Project
    :param client:
    :param section_id: Section ID == MDTA.project.module
    :return:
    """
    client.send_post('delete_section/' + section_id, None)


def add_testcase_to_section(client, section_id, data):
    """
    Add Testcases to TestRail.Project.TestSuites.Section
    :param client:
    :param section_id: Section Id == MDTA.project.module
    :param data: TestCases
    :return:
    """
    try:
        for each_tc in data:
            if 'tcs_cannot_route' not in each_tc.keys():
                custom_preconds = ''
                for pre_cond in each_tc['pre_conditions']:
                    custom_preconds += pre_cond + '; '

                tc_data = {
                    'title': each_tc['title'],
                    'custom_preconds': custom_preconds,
                    'custom_steps_seperated': each_tc['tc_steps']
                }
                client.send_post('add_case/' + section_id, tc_data)
    except APIError as e:
        print('Add TestCase to Section error: ', e)


# --------------- TestRail End ---------------

