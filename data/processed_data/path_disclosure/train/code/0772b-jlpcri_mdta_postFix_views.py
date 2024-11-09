import socket
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render, redirect
import json


from mdta.apps.projects.models import Project, Module, TestRailInstance, TestRailConfiguration
from mdta.apps.projects.utils import context_project_dashboard
from mdta.apps.testcases.models import TestCaseResults
from mdta.apps.testcases.tasks import create_testcases_celery, push_testcases_to_testrail_celery
from mdta.apps.users.views import user_is_superuser, user_is_staff
from .utils import context_testcases, get_projects_from_testrail, create_routing_test_suite
from .forms import TestrailConfigurationForm
from mdta.apps.testcases.testrail import APIClient
from mdta.celery_module import app as celery_app


@login_required
def tcs_project(request):
    if request.user.humanresource.project:
        project = request.user.humanresource.project
        try:
            testcases = project.testcaseresults_set.latest('updated').results
        except TestCaseResults.DoesNotExist:
            testcases = []
    else:
        testcases = []
        project = None

    context = {
        'project': project,
        'testcases': testcases
    }

    if project:
        return render(request, 'testcases/tcs_project.html', context)
    else:
        return redirect('graphs:projects_for_selection')


@user_passes_test(user_is_superuser)
def testcases(request):
    context = context_testcases()

    return render(request, 'testcases/testcases.html', context)


@user_passes_test(user_is_staff)
def create_testcases(request, object_id):
    """
    Create TestCases per project/module
    :param request:
    :param object_id: project_id/module_id
    :return:
    """

    testcases = []
    link_id = ''
    level = request.GET.get('level', '')
    if level == 'project':
        <fix/># create_testcases_celery(object_id, call_from='OldTC')
        project = get_object_or_404(Project, pk=object_id)
        testcases = create_routing_test_suite(project)</fix>

    elif level == 'module':
        module = get_object_or_404(Module, pk=object_id)
        link_id = module.project.id
        testcases = create_routing_test_suite(modules=[module])

    context = context_testcases()
    context['testcases'] = testcases
    context['link_id'] = link_id

    return render(request, 'testcases/testcases.html', context)


@user_passes_test(user_is_superuser)
def create_testcases_all(request):
    projects = Project.objects.all()
    for project in projects:
        create_testcases_celery.delay(project.id)

    return redirect('testcases:testcases')


@login_required
def demonstrate_testcases(request, object_id):
    """
    Demonstrate TestCases of Project/Module from TestCaseResults
    :param request:
    :param object_id:
    :return:
    """
    level = request.GET.get('level', '')
    if level == 'project':
        project = get_object_or_404(Project, pk=object_id)
        link_id = project.id
        try:
            testcases = project.testcaseresults_set.latest('updated').results
        except TestCaseResults.DoesNotExist:
            testcases = []
    elif level == 'module':
        module = get_object_or_404(Module, pk=object_id)
        link_id = module.project.id
        try:
            tmp_tcs = module.project.testcaseresults_set.latest('updated').results
            testcases = [(item for item in tmp_tcs if item['module'] == module.name).__next__()]
        except TestCaseResults.DoesNotExist:
            testcases = []
    else:
        testcases = []
        link_id = ''

    context = context_testcases()
    context['testcases'] = testcases
    context['link_id'] = link_id

    return render(request, 'testcases/testcases.html', context)


@user_passes_test(user_is_staff)
def push_testcases_to_testrail(request, project_id):
    """
    Push Testcases of project to TestRail
    :param request:
    :param project_id:
    :return:
    """
    testrail_contents = push_testcases_to_testrail_celery.delay(project_id)
    # testrail_contents = push_testcases_to_testrail_celery.delay(project_id)

    context = context_testcases()
    context['testrail'] = testrail_contents
    context['link_id'] = project_id

    return render(request, 'testcases/testcases.html', context)


@user_passes_test(user_is_staff)
def testrail_configuration_new(request):
    if request.method == 'GET':
        context = {
            'form': TestrailConfigurationForm()
        }
        return render(request, 'testcases/tc_testrails_new.html', context)
    elif request.method == 'POST':
        # print(request.POST)

        instance = get_object_or_404(TestRailInstance, username='testrail@west.com')
        testrail_projects = get_projects_from_testrail(instance)

        form = TestrailConfigurationForm(request.POST)
        if form.is_valid():
            suites = []
            testrail_new = form.save(commit=False)
            testrail_find = next(item for item in testrail_projects if item['name'] == testrail_new.project_name)
            testrail_new.project_id = testrail_find['id']

            client = APIClient(testrail_new.instance.host)
            client.user = testrail_new.instance.username
            client.password = testrail_new.instance.password
            testrail_find_suites = client.send_get('get_suites/' + str(testrail_new.project_id))
            for suite in testrail_find_suites:
                suites.append(suite['name'])
            testrail_new.test_suite = suites

            testrail_new.save()
        else:
            messages.error(request, form.errors)

        context = context_project_dashboard(request)
        context['last_tab'] = 'test_rails'

        return render(request, 'projects/project_dashboard.html', context)


@user_passes_test(user_is_superuser)
def testrail_configuration_delete(request, testrail_id):
    testrail = get_object_or_404(TestRailConfiguration, pk=testrail_id)

    testrail.delete()

    context = context_project_dashboard(request)
    context['last_tab'] = 'test_rails'

    return render(request, 'projects/project_dashboard.html', context)


@user_passes_test(user_is_superuser)
def testrail_configuration_update(request, testrail_id):
    suites = []
    testrail = get_object_or_404(TestRailConfiguration, pk=testrail_id)

    client = APIClient(testrail.instance.host)
    client.user = testrail.instance.username
    client.password = testrail.instance.password
    testrail_find_suites = client.send_get('get_suites/' + str(testrail.project_id))
    for suite in testrail_find_suites:
        suites.append(suite['name'])

    if testrail.test_suite != suites:
        testrail.test_suite = suites
        testrail.save()

    context = context_project_dashboard(request)
    context['last_tab'] = 'test_rails'

    return render(request, 'projects/project_dashboard.html', context)


def check_celery_task_state(request):
    task_run = False
    active = celery_app.control.inspect().active()

    # celery worker node name
    key = 'celery@' + socket.gethostname() + '.mdta'
    try:
        if active[key]:
            project_id = active[key][0]['args']
            project_id = ''.join(c for c in project_id if c not in '\'(),')
            if int(project_id) == request.user.humanresource.project.id:
                task_run = True
    except (KeyError, TypeError):
        task_run = True

    return HttpResponse(json.dumps(task_run), content_type='application/json')


