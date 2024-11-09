from rest_framework import serializers
from rest_framework.validators import ValidationError

from .models import Message


class MessageSerializer(serializers.ModelSerializer):
    """
    Message Serializer class
    """

    class Meta:
        model = Message
        fields = ('sent_by', 'room', 'text', 'attachment')
