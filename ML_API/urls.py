from django.urls import path

from . import views

urlpatterns = [
    #View Classes go here:
    #Ex: if ListTodo was a class in views.py then:
    #path('', views.ListTodo.as_view()),
    path('', views.getRoutes, name='routes'),
    path('saveCategory', views.saveCategory, name='saveCategory'),
    path('saveDBImage', views.saveDBImageLink, name='saveDBImageLink'),
    path('showDBImage', views.showDBImage, name='showDBImage'),
    path('trainCV', views.trainCat, name='train'),
    path('completeTrainCV', views.completeTrain, name='completeTrainCV'),
    path('predictCV', views.predict, name='predict'),
    path('deleteDBImage', views.deleteCategory, name='deleteDBImage'),
    path('deleteCategoryClassifier', views.deleteCategoryClassifier, name='deleteCategoryClassifier')
]