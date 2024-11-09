from django.shortcuts import render
from django.db import transaction
from django.core import serializers
from django.http import HttpResponse

from sio.models import *
from sio.forms import *

def make_view(request,
              messages=[],
              create_student_form=CreateStudentForm(),
              create_course_form=CreateCourseForm(),
              register_student_form=RegisterStudentForm()):
    context = {
               'courses':Course.objects.all(),
               'messages':messages,
               'create_student_form':create_student_form,
               'create_course_form':create_course_form,
               'register_student_form':register_student_form,
              }
    return render(request, 'sio.html', context)

def home(request):
    return make_view(request, [])

@transaction.atomic
def create_student(request):
    form = CreateStudentForm(request.POST)
    if not form.is_valid():
        return make_view(request, create_student_form=form)

    new_student = Student(andrew_id=form.cleaned_data['andrew_id'],
                          first_name=form.cleaned_data['first_name'],
                          last_name=form.cleaned_data['last_name'])
    new_student.save()
    return make_view(request, ['Added %s'%new_student])

@transaction.atomic
def create_course(request):
    form = CreateCourseForm(request.POST)
    if not form.is_valid():
        return make_view(request, create_course_form=form)

    new_course = Course(course_number=request.POST['course_number'],
                        course_name=request.POST['course_name'],
                        instructor=request.POST['instructor'])
    new_course.save()
    return make_view(request, messages=['Added %s'%new_course])

@transaction.atomic
def register_student(request):
    form = RegisterStudentForm(request.POST)
    if not form.is_valid():
        return make_view(request, register_student_form=form)

    course = Course.objects.get(course_number=request.POST['course_number'])
    student = Student.objects.get(andrew_id=request.POST['andrew_id'])
    course.students.add(student)
    course.save()
    return make_view(request, messages=['Added %s to %s' % (student, course)])


def get_student_by_name(request):
    first_name = request.GET.get('first_name', '')

    # The normal use of the ORM to get students by first name:
    #students = Student.objects.filter(first_name__exact=first_name)

    # The correct way to use raw SQL to get students by first name:
    <vul/>#students = Student.objects.raw('select * from uni_student where first_name = %s', [first_name])</vul>

    # Gets students by first name, but is vulnerable to SQL injection attacks:
    <vul/>students = Student.objects.raw('select * from sio_student where first_name = \'' + first_name + '\'')</vul>

    response_text = serializers.serialize('json', students)
    return HttpResponse(response_text, content_type='application/json')

