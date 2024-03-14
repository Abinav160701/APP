import streamlit as st 
import pickle
import os
from io import BytesIO
import requests
import numpy as np
import tensorflow as tf
from annoy import AnnoyIndex
import pandas as pd
from detectron2.change_bg import get_bg_changed
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
#import cv2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from Preprocess import process_query_image_mc
from models import load_model_from_db
from load_feature_vec import load_features_from_db
from Opensearchfn import get_similar_skus




# Set the app title 
st.title('Image Matching App') 
# Add a welcome message 
st.write('Please upload a test image!') 
# Create a text input 
# widgetuser_input = st.text_input('Enter a custom message:', 'Hello, Streamlit!') 
# Display the customized message 
# st.write('Customized Message:', widgetuser_input)

with st.sidebar:
    # st.[element_name]
    l1 = st.selectbox('Choose l1 category',('Men', 'Women', 'Kids'))
    l2 = st.selectbox('Choose l2 category',('Clothing', 'Shoes', 'Bags'))
    picture=None
    #picture = st.camera_input("Take a picture") 
    if picture:
        test_image=picture
    else:
        test_image = st.file_uploader("Choose a file")
    if test_image is not None:
        # To read file as bytes:
        # bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)

        st.image(test_image,width = 200, caption='Test Image')


model = load_model_from_db('Men','Clothing')

features,labels= load_features_from_db(l1,l2)



    

# st.write(loaded_image_paths)

# st.write('loaded features')
def get_image(d):
    try:
        url = d.loc[0,'img1']
        r = requests.get(url)
        return BytesIO(r.content)
    except:
        return "https://www.shutterstock.com/image-vector/default-avatar-profile-icon-social-media-1677509740"
    

def find_sim_images(query_features,df):
    
   #similarities = cosine_similarity(query_features, np.array(features))
   #top_indices = np.argsort(similarities[0])[::-1][:]   Adjust the number as needed
    skus=get_similar_skus(query_features)
    l=[]
    st.write(skus[0])
    for i in skus:
        l.append(i)
    d=df.query(f"sku in {l}")
    for i in range(10):
        st.write("class:",skus[i][:])
        st.image(get_image(df.query(f"sku=='{skus[i][:]}'").reset_index(drop=True)),width=100)
    '''l=[]
#    for i in top_indices:
#        if labels[i]=='TH6-MW0MW307050A4':
#            st.write(np.where(top_indices==i))
    top_indices_unique=[]
    labels_unique=[]
    for i in top_indices:
        if labels[i] not in labels_unique:
            labels_unique.append(labels[i])
            top_indices_unique.append(i)
#    for i in top_indices:
#        if i not in top_indices_unique:
#            top_indices_unique.append(i)
#            l.append(labels[i])'''
    '''d=df.query(f"sku in {labels_unique}")
    for i in top_indices_unique[:20]:
        if '/' in labels[i]:
            st.write("class:",labels[i][:-12],"similarity:",similarities[0][i])
            x=labels[i][:-12]
        else:
            st.write("class:",labels[i][:],"similarity:",similarities[0][i])
            x=labels[i]
        #st.image(get_image(d.query(f"sku=='{x}'").reset_index(drop=True)))
        st.image(get_image(df.query(f"sku=='{x}'").reset_index(drop=True)),width=100)'''
    



def find_similar_images_annoy(query_features, annoy_index, labels,df, top_n=20):
    similarities = cosine_similarity(query_features, np.array(features))
    top_indices = np.argsort(similarities[0])[::-1][:40]  # Adjust the number as needed
    # Convert distances to similarity scores
    l=0
    unique_labels=[]
    for i in top_indices:
        unique_labels.append(labels[i])
    d=df.query(f"sku in {unique_labels}")
    for i in top_indices:
        st.write("class:",labels[i],"Similarity:",similarities[l])
        l+=1
        st.image(get_image(d.query(f"sku=='{labels[i]}'").reset_index(drop=True)),width=100)


    #st.write(len(labels))
    st.write(len(similarities))
    
    #return [(labels[i], similarities[i]) for i in indices]



# def preprocess_image_uploaded_img(image_path):
#     # Load and preprocess an image for EfficientNet
#     img = image.load_img(test_image, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
#     return img_array

# def extract_features_uploaded_img(model, image_path):
#     # Extract features using the EfficientNet model
#     img_array = preprocess_image(image_path)
#     features = model.predict(img_array)
#     return features.flatten()

df=pd.read_csv(f'csv_db/{l1}_{l2}_db.csv')
df=df.iloc[:,1:5]
df.columns=['sku','img1','img2','img3']

if test_image is not None:
    #model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    # query_image_path = 'query/img_test1.jpeg'
    # Load the saved features and model
    query_image_path = test_image
    print(query_image_path)
    #white_img=get_bg_changed(query_image_path)
    query_img = process_query_image_mc(query_image_path)
    query_features = model.predict(query_img)
    # st.write(query_features)
    # Load the Annoy index

    find_sim_images(query_features,df)
    #find_similar_images_annoy(query_features[0], annoy_index, labels,df, top_n=20)
    #similar_images=find_similar_images_annoy(query_features[0], annoy_index, labels)
    #display_annoy_results(similar_images,df)
    # Similarity search
    #similarities = cosine_similarity(query_features, np.array(features))
    #top_indices = np.argsort(similarities[0])[::-1][:20]  # Adjust the number as needed
else :
    st.write('waiting for test image....')



