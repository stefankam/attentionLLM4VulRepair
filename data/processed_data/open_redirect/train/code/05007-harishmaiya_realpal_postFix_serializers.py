from rest_framework import serializers

<fix/>from .models import Message, Room</fix>


class MessageSerializer(serializers.ModelSerializer):
    """
    Message Serializer class
    """

    class Meta:
        model = Message
        fields = ('sent_by', 'room', 'text', 'attachment')


class RoomSerializer(serializers.ModelSerializer):
    """
    Room Serializer class
    """

    class Meta:
        model = Room
        fields = ('id', 'client', 'agent')
        read_only_fields = ('client',)
