import logging
import json
import os
from urllib.parse import urlparse

from channels import Group
<fix/>from rest_framework.generics import CreateAPIView, UpdateAPIView</fix>
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status


from django.conf import settings
<fix/>from django.core.urlresolvers import reverse

from realpal.apps.chat.serializers import MessageSerializer, RoomSerializer</fix>
from realpal.apps.chat.consumers import get_room_group_channel
<fix/>from realpal.apps.chat.models import Message, Room</fix>

logger = logging.getLogger(__name__)


class MessageCreateAPIView(CreateAPIView):
    """
    Creates a new message object with a file attachment

    Returns on the socket

        {
            'id': "id",
            'sent_by':'user_id',
            'room':"room_id",
            'text':message.txt,
            'file_name': message.attachment,
            'file_link': message.attachment.path
        }
    """
    model = Message
    serializer_class = MessageSerializer
    permission_classes = [IsAuthenticated, ]

    def create(self, request, *args, **kwargs):
        room_id = self.request.data.get('room')
        try:
            self.room = Room.objects.get(pk=room_id)
            self.request.data['sent_by'] = self.request.user.id
            self.request.data['room'] = self.room.id
            self.request.data['text'] = self.request.data.get('message')
            serializer = self.get_serializer(data=request.data)
            self.perform_create(serializer)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Room.DoesNotExist:
            return Response(status=status.HTTP_400_BAD_REQUEST)

    def perform_create(self, serializer):
        serializer.is_valid(self)
        instance = serializer.save(sent_by=self.request.user, room=self.room)
        if not settings.IS_TESTING:
            data = {
                'id': instance.id.__str__(),
                'timestamp': instance.time_ago,
                'timestamp_string': instance.timestamp_string,
                'user_handle': self.request.user.full_name,
                'user_type': self.request.user.user_type,
                'message': instance.text,
                'file_name': os.path.basename(urlparse(instance.attachment.path).path) if instance.attachment else None,
                'file_link': instance.file_download_link if instance.attachment else None,
            }
            group_channel = get_room_group_channel(instance.room.id)
            self.push_socket_update(group_channel, data)

    @staticmethod
    def push_socket_update(group_channel, data):
        Group(group_channel).send({"text": json.dumps(data)})


class RoomUpdateAPIView(UpdateAPIView):
    model = Room
    serializer_class = RoomSerializer
    permission_classes = [IsAuthenticated, ]

    def patch(self, request, *args, **kwargs):
        room_id = self.request.data.get('id', 0)
        try:
            room = Room.objects.get(pk=room_id)
            if room.agent == self.request.user:
                room.agent = None
                room.save()
                data = {
                    'client_room_id': room.id,
                    'client_room_link': reverse('chat:chat-room', args=(room.id,)),
                    'client_details': room.client.full_name,
                }
                return Response(data=data, status=status.HTTP_200_OK)
            else:
                return Response(status=status.HTTP_403_FORBIDDEN)
        except Room.DoesNotExist:
            return Response(status=status.HTTP_400_BAD_REQUEST)


