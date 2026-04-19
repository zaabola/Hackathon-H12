from rest_framework import serializers
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import CustomUser


class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    """Adds user role and info to the JWT token response."""

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['role'] = user.role
        token['username'] = user.username
        token['email'] = user.email
        return token

    def validate(self, attrs):
        data = super().validate(attrs)
        data['role']     = self.user.role
        data['username'] = self.user.username
        data['email']    = self.user.email
        data['id']       = self.user.id
        data['is_approved'] = self.user.is_approved
        return data


class UserSerializer(serializers.ModelSerializer):
    """Full serializer used by admin CRUD endpoints."""
    password         = serializers.CharField(write_only=True, required=False)
    id_document_url  = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = CustomUser
        fields = [
            'id', 'username', 'email', 'first_name', 'last_name',
            'role', 'department', 'phone', 'password',
            'is_active', 'is_approved', 'rejection_note',
            'id_document', 'id_document_url',
            'date_joined',
        ]
        read_only_fields = ['id', 'date_joined', 'id_document_url']
        extra_kwargs = {
            'id_document': {'write_only': True},
        }

    def get_id_document_url(self, obj):
        request = self.context.get('request')
        if obj.id_document and request:
            return request.build_absolute_uri(obj.id_document.url)
        if obj.id_document:
            return obj.id_document.url
        return None

    def create(self, validated_data):
        password = validated_data.pop('password', None)
        user = CustomUser(**validated_data)
        if password:
            user.set_password(password)
        else:
            user.set_unusable_password()
        user.save()
        return user

    def update(self, instance, validated_data):
        password = validated_data.pop('password', None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance


class FarmerRegisterSerializer(serializers.ModelSerializer):
    """Public self-registration serializer for farmers only."""
    password    = serializers.CharField(write_only=True, min_length=6)
    id_document = serializers.ImageField(required=True)

    class Meta:
        model = CustomUser
        fields = [
            'username', 'email', 'password',
            'first_name', 'last_name', 'phone',
            'id_document',
        ]

    def validate_username(self, value):
        if CustomUser.objects.filter(username=value).exists():
            raise serializers.ValidationError('Username already taken.')
        return value

    def validate_email(self, value):
        if value and CustomUser.objects.filter(email=value).exists():
            raise serializers.ValidationError('Email already registered.')
        return value

    def create(self, validated_data):
        password = validated_data.pop('password')
        user = CustomUser(
            **validated_data,
            role='farmer',
            is_active=False,      # blocks login until approved
            is_approved=False,    # admin must approve
        )
        user.set_password(password)
        user.save()
        return user
