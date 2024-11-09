<fix/>from django.shortcuts import render, get_object_or_404</fix>

from .models import Post

# Create your views here.
def home(request):
    <fix/>posts = Post.objects.order_by('-pub_date')</fix>
    return render(request, 'posts/home.html', {'posts':posts})

def post_details(request, post_id):
    <fix/>post = get_object_or_404(Post, pk=post_id)
    return render(request, 'posts/posts_detail.html', {'post':post})</fix>
