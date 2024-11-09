<vul/>from django.shortcuts import render</vul>

from .models import Post

# Create your views here.
def home(request):
    <vul/>posts = Post.objects.order_by('pub_date')</vul>
    return render(request, 'posts/home.html', {'posts':posts})

def post_details(request, post_id):
    <vul/>return render(request, 'posts/posts_detail.html', {'post_id':post_id})</vul>
