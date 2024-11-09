from django.contrib import admin
from django.urls import re_path
from Shortener_App.views import (
    HomeView,
    SuccessUrlView,
    CustomShortURLCreateView,
    ShortManyURLSView,
    URLDetailView,
    URLUpdateView,
    URLDeleteView,
    CategoryCreateView,
    CategoryListView,
    CategoryDetailView,
    CategoryUpdateView,
    CategoryDeleteView,
    ClickTrackingDetailView,
    link_redirect
)

urlpatterns = [
    re_path(r'admin/', admin.site.urls),
    #  urls
    re_path(r'^$', HomeView.as_view(), name='home-view'),
    re_path(r'^success/(?P<pk>(\d)+)/$', SuccessUrlView.as_view(), name='success-url-view'),
    re_path(r'^add-custom/$', CustomShortURLCreateView.as_view(), name='add-custom-url'),
    re_path(r'^add-many/$', ShortManyURLSView.as_view(), name='add-many-urls'),
    re_path(r'^detail/(?P<pk>(\d)+)/$', URLDetailView.as_view(), name='url-detail-view'),
    re_path(r'^update/(?P<pk>(\d)+)/$', URLUpdateView.as_view(), name='url-update-view'),
    re_path(r'^delete/(?P<pk>(\d)+)/$', URLDeleteView.as_view(), name='url-delete-view'),
    #  categories
    re_path(r'^category/add/$', CategoryCreateView.as_view(), name='category-create-view'),
    re_path(r'^categories/$', CategoryListView.as_view(), name='category-list-view'),
    re_path(r'^detail/category/(?P<pk>(\d)+)/$', CategoryDetailView.as_view(), name='category-detail-view'),
    re_path(r'^update/category/(?P<pk>(\d)+)/$', CategoryUpdateView.as_view(), name='category-update-view'),
    re_path(r'^delete/category/(?P<pk>(\d)+)/$', CategoryDeleteView.as_view(), name='category-delete-view'),
    #  click tracking
    re_path(r'^(?P<pk>(\d)+)/reports/$', ClickTrackingDetailView.as_view(), name='clicktracking-detail-view'),
    #  redirecting
    <fix/>re_path(r'^(?P<token>(\w)+)/$', link_redirect, name='url-redirect-view'),</fix>
]
