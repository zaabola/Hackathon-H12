from rest_framework import serializers
from .models import IncidentLog
from accounts.serializers import UserSerializer


class IncidentLogSerializer(serializers.ModelSerializer):
    created_by_detail = UserSerializer(source='created_by', read_only=True)
    labeled_by_detail = UserSerializer(source='labeled_by', read_only=True)
    photo_url = serializers.SerializerMethodField()
    video_url = serializers.SerializerMethodField()

    class Meta:
        model = IncidentLog
        fields = [
            'id', 'module', 'source', 'title', 'description',
            'photo', 'photo_url', 'video', 'video_url',
            'result_data', 'zone_name', 'latitude', 'longitude',
            'status', 'labeled_by', 'labeled_by_detail', 'label_note', 'labeled_at',
            'created_by', 'created_by_detail', 'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'created_by', 'labeled_by', 'labeled_at', 'created_at', 'updated_at']

    def get_photo_url(self, obj):
        if obj.photo:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.photo.url)
            return obj.photo.url
        return None

    def get_video_url(self, obj):
        if obj.video:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.video.url)
            return obj.video.url
        return None

    def create(self, validated_data):
        validated_data['created_by'] = self.context['request'].user
        return super().create(validated_data)


class LabelLogSerializer(serializers.ModelSerializer):
    """Serializer for technicians to label a log."""

    class Meta:
        model = IncidentLog
        fields = ['status', 'label_note']

    def update(self, instance, validated_data):
        from django.utils import timezone
        instance.status = validated_data.get('status', instance.status)
        instance.label_note = validated_data.get('label_note', instance.label_note)
        instance.labeled_by = self.context['request'].user
        instance.labeled_at = timezone.now()
        instance.save()
        return instance


class LogStatsSerializer(serializers.Serializer):
    total = serializers.IntegerField()
    in_progress = serializers.IntegerField()
    resolved = serializers.IntegerField()
    non_resolved = serializers.IntegerField()
    by_module = serializers.DictField()
