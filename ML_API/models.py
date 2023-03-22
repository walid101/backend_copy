from django.db import models

class ImageModel(models.Model):
    img = models.ImageField(null=True, blank=True, upload_to='images/')

class TrainImages(models.Model):
    TrainImageId=models.AutoField(primary_key=True)
    className = models.CharField(max_length=255) #This will be the 

class PredictImage(models.Model):
    url=models.CharField(max_length=255)
    image=models.ImageField(upload_to ='uploads/', blank=True, null=True)
    categories=models.CharField(max_length=255, blank=True, null=True)

class TrainRequest(models.Model):
    category = models.CharField(max_length=255, blank=True) #This will be the class (ex: 'bird') 
                                                #that we need to train our model for,
                                                #we will retrive images with google 
                                                #scraper
    population = models.IntegerField(blank=True) #Number of images to train on
    
class DBImage(models.Model):
    category = models.CharField(max_length=255)
    image_name = models.CharField(max_length=255)
    image_link = models.CharField(max_length=255)

    