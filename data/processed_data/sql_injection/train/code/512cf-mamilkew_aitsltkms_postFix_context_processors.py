from django.utils import timezone
from django.db.models import Avg, Count, Min, Max, Sum


def navbarmain(request):
    from pages.models import Post
    <fix/>raw_query = "SELECT DISTINCT ON (subject) *, max(published_date) FROM pages_post WHERE published_date is not null GROUP BY subject, id ORDER BY subject ASC , published_date DESC"
    # for p in Post.objects.raw(raw_query):
    #     print(p)
    # pb = Post.objects.values('subject').annotate(newest_published_date=Max('published_date'))
    # pb_list = Post.objects.filter(subject__in=[b.get('subject') for b in pb], published_date__in=[b.get('newest_published_date') for b in pb])</fix>
    # print(pb_list)
    <fix/>return {'navbarMain': Post.objects.raw(raw_query)}</fix>
