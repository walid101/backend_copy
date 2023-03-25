import random
from django.shortcuts import render
from django.db import models
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
from rest_framework.parsers import JSONParser
from django.http.response import JsonResponse


from pymongo import MongoClient
from ML_API.models import TrainImages, PredictImage, TrainRequest, DBImage
from ML_API.serializers import TrainImagesSerializer, PredictImageSerializer, TrainRequestSerializer, DBImageSerializer
from PIL import Image
import io, lxml
import numpy as np


import skimage.exposure as exposure
from skimage.transform import resize
from skimage.feature import hog
from skimage import io as skio
from sklearn.svm import LinearSVC, SVC   
from sklearn.calibration import CalibratedClassifierCV
# from cv2 import cvtColor, COLOR_BGR2RGB
import re, json, pickle, dill
from bs4 import BeautifulSoup as bs

client = MongoClient("mongodb+srv://dbuser:YSHHbWgldJNQ3xIE@cluster0.1otrah0.mongodb.net/?retryWrites=true&w=majority")
db = client["LogBook"]

def getRoutes(request):
	print(request)
	return JsonResponse('Our API', safe=False)

def saveImage(img_url):
    return ''

def scrape_imgs(queries):

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36",
    }

    google_images = []
    for query in queries:
        print(f'Extracting images for query: {query}')
        
        params = {    
            "q": "real " + query,              # search query
            "tbm": "isch",                                   # image results
            "hl": "en",                                      # language of the search
            "gl": "us",                                      # country where search comes fro
        }
        
        html = requests.get("https://google.com/search", params=params, headers=headers, timeout=30)
        soup = bs(html.text, "lxml")
        
        all_script_tags = soup.select("script")
        
        matched_images_data = "".join(re.findall(r"AF_initDataCallback\(([^<]+)\);", str(all_script_tags)))
        
        matched_images_data_fix = json.dumps(matched_images_data)
        matched_images_data_json = json.loads(matched_images_data_fix)
        
        matched_google_image_data = re.findall(r'\"b-GRID_STATE0\"(.*)sideChannel:\s?{}}', matched_images_data_json)

        matched_google_images_thumbnails = ", ".join(
            re.findall(r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]',
                        str(matched_google_image_data))).split(", ")
        
        thumbnails = [bytes(bytes(thumbnail, "ascii").decode("unicode-escape"), "ascii").decode("unicode-escape") for thumbnail in matched_google_images_thumbnails]
        
        # removing previously matched thumbnails for easier full resolution image matches.
        removed_matched_google_images_thumbnails = re.sub(
                r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]', "", str(matched_google_image_data))
        
        matched_google_full_resolution_images = re.findall(r"(?:'|,),\[\"(https:|http.*?)\",\d+,\d+\]", removed_matched_google_images_thumbnails)
        
        full_res_images = [
                bytes(bytes(img, "ascii").decode("unicode-escape"), "ascii").decode("unicode-escape") for img in matched_google_full_resolution_images
        ]
            
        for index, (metadata, thumbnail, original) in enumerate(zip(soup.select('.isv-r.PNCib.MSM1fd.BUooTd'), thumbnails, full_res_images), start=1):
            google_images.append({
                "title": metadata.select_one(".VFACy.kGQAp.sMi44c.lNHeqe.WGvvNb")["title"],
                "link": metadata.select_one(".VFACy.kGQAp.sMi44c.lNHeqe.WGvvNb")["href"],
                "source": metadata.select_one(".fxgdke").text,
                "thumbnail": thumbnail,
                "original": original
            })

        #print(json.dumps(google_images, indent=2, ensure_ascii=False))
        return thumbnails

@csrf_exempt
def deleteCategory(request):
     dbimg_request=JSONParser().parse(request)
     dbimg_request_serializer=DBImageSerializer(data=dbimg_request)
     if(dbimg_request_serializer.is_valid()):
          images = db.images
          cat = dbimg_request['category']
          images.remove({'category': dbimg_request['category']})
          return JsonResponse("Successfully Deleted Category", safe=False)
     return JsonResponse("Deletion Failed")

@csrf_exempt
def deleteCategoryClassifier(request):
     db_request=JSONParser().parse(request)
     db_request_serializer=TrainRequestSerializer(data=db_request)
     if(db_request_serializer.is_valid()): 
          clfs = db.classifiers
          cat = db_request['category']
          if(cat == "DeleteAll"):
               clfs.delete_many({})
          else:
               clfs.delete_one({"category":cat})
          return JsonResponse("Successfully Deleted: " + cat, safe=False)
     return JsonResponse("Invalid Request: DeleteCategoryClassifier", safe=False)

@csrf_exempt
def saveDBImageLink(request):
     print(request.body)
     dbimg_request=JSONParser().parse(request)
     dbimg_request_serializer=DBImageSerializer(data=dbimg_request)
     if dbimg_request_serializer.is_valid():
          images = db.images
          image_bytes = io.BytesIO()
          im = Image.open(requests.get(dbimg_request["image_link"], stream = True).raw)
          
          im.save(image_bytes, format='JPEG')
          image = {
               'category' : dbimg_request["category"],
               'name': dbimg_request["image_name"],
               'data': image_bytes.getvalue()
          }
          images.insert_one(image)
          return JsonResponse("Test Image Sent Successfully!", safe=False)
     print(dbimg_request_serializer.errors)
     return JsonResponse("Invalid Request", safe=False)

@csrf_exempt
def saveCategory(request):
     print(request.body)
     train_request = JSONParser().parse(request)
     train_request_serializer = TrainRequestSerializer(data=train_request)
     if train_request_serializer.is_valid():
          #scrape images here and save to db
          category = train_request["category"]
          pop = train_request["population"]
          links = scrape_imgs([category])[0:pop]
          loadData(links=links, cat_name=category if category != "random objects" else "Default", save=True)
          return JsonResponse("Category Images Sent Successfully!", safe=False)
     print(train_request_serializer.errors)
     return JsonResponse("Invalid Request", safe=False)

def saveImage(im, im_name, category):
     #im is an Image object already
     images = db.images
     image_bytes = io.BytesIO()
     try:
          im.save(image_bytes, format='JPEG')
     except:   
          return

     image = {
       'category' : category,
       'name': im_name,
       'data': image_bytes.getvalue()
     }

     image_id = images.insert_one(image).inserted_id
     return image_id

def saveClassifier(category, clf):
     pickled_clf = None
     try:
          pickled_clf = pickle.dumps(clf)
     except:
          pickled_clf = dill.dumps(clf)
     clfs = db.classifiers
     db_clf = {
          'category': category,
          'model': pickled_clf
     }
     if(clfs.find_one({"category":category}) is None):
          clfs.insert_one(db_clf)
     else:
          clfs.replace_one({"category":category}, db_clf)
     return JsonResponse("Classifier Saved Successfully!", safe=False)

def getClassifier(category):
     clfs = db.classifiers
     model = pickle.loads(clfs.find_one({'category': category})['model'])
     print(model)
     return model

def Categories():
     clfs = db.classifiers
     return clfs.count_documents({}), clfs.find({})

def getUniformRandomDistCats(excluded_cat=["Default"], random_size = 32):
     print("Getting Uniform Random Dist")
     collection = db.images
     categories = collection.distinct("category")
     numCats, Cats = Categories()
     unif_size = 0
     if(numCats > 0):
          unif_size = int(random_size/numCats)
     unif_random_cats = []
     for category in categories:
          if(category in excluded_cat):
               continue
          curr_cat = getImages(name=None, cat=category, classNum=0)[0:unif_size]
          unif_random_cats.extend(curr_cat)
     random.shuffle(unif_random_cats)
     return unif_random_cats

@csrf_exempt
def showDBImage(request, id=2):
     dbimg_request=JSONParser().parse(request)
     dbimg_request_serializer=DBImageSerializer(data=dbimg_request)
     if dbimg_request_serializer.is_valid():
          images = db.images
          test_imgs = images.find({"category" : dbimg_request["image_name"]})[0:5] #limiting to 5 images for now
          for ima in test_imgs:
               image = Image.open(io.BytesIO(ima['data']))
               image.show()
          return JsonResponse("Successfully Displayed Test Image", safe=False)
     print(dbimg_request_serializer.errors)
     return JsonResponse("Invalid Request", safe=False)

def getImages(name=None, cat=None, classNum = 1):
     images = db.images
     imgs = None

     if(cat != None):
          imgs = images.find({"category":cat})
     elif(name != None):
          imgs = images.find({"name" : name})
     processed_imgs = []
     for img in imgs:
          try:
               image = Image.open(io.BytesIO(img['data']))
          except:
               continue
          image = [np.array(image), 0 if cat == "Default" else classNum]
          processed_imgs.append(image)
     return processed_imgs

def loadData(links, cat_name="Default", save=False):
     #extract image from link and save it to db as IOBytes
     #return list of images from links
     imgs = []
     for link in links:
          img = Image.open(requests.get(link, stream = True).raw)
          imgs.append([np.array(img), 0 if cat_name == "Default" else 1])
          if(save):
               saveImage(im = img, im_name = cat_name if cat_name != None else "Default", category=cat_name)
     return imgs

@csrf_exempt #Only train if classifier is NOT found
def trainCat(request, cat=None):
     train_request=cat if cat is not None else JSONParser().parse(request)
     train_request_serializer=cat if cat is not None else TrainRequestSerializer(data=train_request)
     print("Train Start: ", train_request)
     if cat is not None or train_request_serializer.is_valid():
          #Scrape Category Images and Train Model here
          category = cat if cat is not None else train_request["category"]
          cat_images = getImages(cat=category) #from db
          ran_images = getUniformRandomDistCats(excluded_cat=["Default", category]) #from db, attempt to get an even random collection of all other categories first, fill any space with random objects
          if(len(cat_images) == 0):
               cat_images.extend(loadData(scrape_imgs([category])[0:32-len(cat_images)], cat_name=category, save=True)) #32 category images
          if(len(ran_images) == 0):     
               ran_images.extend(loadData(scrape_imgs(["random object"])[0:32-len(ran_images)], cat_name="Default", save=True)) #add random images

          #send images to train -> cat, ran
          classifier, cat = train(cat_name=category, cat_imgs=cat_images, ran_imgs=ran_images)
          
          #save classifier under category name
          saveClassifier(category, classifier)

          return JsonResponse("Successfully Trained Model!", safe=False)
     return JsonResponse("Invalid Request", safe=False)

@csrf_exempt
def completeTrain(request):
     print("Retrain + Train Start")
     train_request=JSONParser().parse(request)
     train_request_serializer=TrainRequestSerializer(data=train_request)
     if train_request_serializer.is_valid():
          num_clfs, clfs = Categories()
          cats = [clf["category"] for clf in clfs]
          if(len(train_request["category"]) > 0):
               cats.insert(0, train_request["category"])
     for cat in cats:
          trainCat(request=None, cat=cat)
     return JsonResponse("Successfully Trained Model!", safe=False)

@csrf_exempt
def predict(request):
     print(request.body)
     pred_request=JSONParser().parse(request)
     pred_request_serializer=PredictImageSerializer(data=pred_request)
     #request contains image-url or image or none, match models to request
     if(pred_request_serializer.is_valid()):
          url = pred_request['url']
          avail_cats = pred_request['categories'].split(",") #make sure this categories is a comma seperated list
          print(avail_cats)
          req_image = None#pred_request['image']
          image = None
          if(url is not None and len(url) > 0):
               image = np.asarray(skio.imread(url))
               pass
          elif(req_image is not None):
               pass #Real Image was provided (Find where it was uploaded)
          else:
               return JsonResponse("No Image Provided")
          db_clfs = db.classifiers
          clfs =  db_clfs.find({})
          max_conf_cat = "Category Not Learned"
          max_conf = 0
          feature, hog_img = computeHOGfeatures(resize(image , (128,64)))
          # plt.figure()
          for clf in clfs:
               cat = clf['category']
               if(avail_cats is not None and len(avail_cats) > 0 and ("all" in avail_cats or cat in avail_cats)):
                    clf = pickle.loads(clf['model'])
                    #clf predict by image hog features, if it is either cat or not, if cat then check conf
                    pred = clf.predict_proba([feature])
                    print(pred, "Category : ", cat)
                    if(pred[0][1] > max_conf):
                         #get conf score (Do Later) #https://medium.com/@manoveg/multi-class-text-classification-with-probability-prediction-for-each-class-using-linearsvc-in-289189fbb100
                         max_conf_cat = cat
                         max_conf = pred[0][1]
          # if(max_conf < 0.5):
          #      max_conf_cat = "Category Not Learned"
          return JsonResponse(max_conf_cat, safe=False)
     return JsonResponse("Invalid Request")

# Compute HOG features for the images
def computeHOGfeatures(image):
    feature, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=2)
    return feature, hog_image
          
def train(cat_name, cat_imgs, ran_imgs):

     concat_cat_ran_classvals = cat_imgs + ran_imgs
     random.shuffle(concat_cat_ran_classvals)
     cat_ran_imgs = [resize(img[0] , (128,64)) for img in concat_cat_ran_classvals] #Pictures must be resized to a 1:2 aspect ratio 

     ## create a vector of labels
     cat_ran_labels = [img[1] for img in concat_cat_ran_classvals]
     # assume labels: bird = 0, human = 1

     # Compute HOG descriptors
     cat_ran_hog_features = []
     cat_ran_hog_imgs = []
     for img in cat_ran_imgs:
          try:
               feature, hog_img = computeHOGfeatures(img)
          except:
               continue
          cat_ran_hog_features.append(feature)
          cat_ran_hog_imgs.append(hog_img)

     # reshape feature matrix
     # Split the data and labels into train and test set
     train_size = 32
     trainsets_hog =  cat_ran_hog_features[0:train_size] #first 16
     trainsets_label = cat_ran_labels[0:train_size]#first 16
     # test_sets = cat_ran_hog_features[train_size:] #test sets 
     
     # train model with SVM
     np_trainsets_hog = np.asarray(trainsets_hog, dtype = np.float32)
     np_trainsets_labels = np.asarray(trainsets_label, dtype = np.int32)

     # call LinearSVC
     svm = LinearSVC()
     clf = CalibratedClassifierCV(svm) 
     # clf = SVC(kernel='linear', probability=True)
     # train SVM
     clf.fit(np_trainsets_hog, np_trainsets_labels)

     return clf, cat_name
    