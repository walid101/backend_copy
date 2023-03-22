from rest_framework import serializers
from ML_API.models import TrainImages, PredictImage, TrainRequest, DBImage

class TrainImagesSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainImages
        fields=('className', 'classImages')

class PredictImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictImage
        fields=('url', 'image', 'categories')

class TrainRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainRequest
        fields=('category', 'population')

class DBImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = DBImage
        fields=('category', 'image_name', 'image_link')