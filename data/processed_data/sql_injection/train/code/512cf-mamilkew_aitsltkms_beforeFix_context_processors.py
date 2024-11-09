from django.utils import timezone
from django.db.models import Avg, Count, Min, Max, Sum


def navbarmain(request):
    from pages.models import Post
    <vul/>navbarMain = Post.objects.filter(published_date__lte=timezone.now())
    # print(Post.objects.values('subject').annotate(newest_published_date=Max('published_date')))
    pb = Post.objects.values('subject').annotate(newest_published_date=Max('published_date'))
    pb_list = Post.objects.filter(subject__in=[b.get('subject') for b in pb], published_date__in=[b.get('newest_published_date') for b in pb])</vul>
    # print(pb_list)
    <vul/>return {'navbarMain': pb_list}</vul>
