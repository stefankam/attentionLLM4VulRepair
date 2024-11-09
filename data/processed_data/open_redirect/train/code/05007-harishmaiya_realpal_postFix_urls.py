from django.conf.urls import url

from realpal.apps.chat.views import ChatRoomView
<fix/>from realpal.apps.chat.api import MessageCreateAPIView, RoomUpdateAPIView</fix>

urlpatterns = [
    url(r'^$', ChatRoomView.as_view(), name='chat-room'),
    url(r'^(?P<room_id>[0-9]+)/', ChatRoomView.as_view(), name='chat-room'),
    url(r'^room/update/$', RoomUpdateAPIView.as_view(), name='update-room'),
    url(r'^file/$', MessageCreateAPIView.as_view(), name='chat-file'),

]
