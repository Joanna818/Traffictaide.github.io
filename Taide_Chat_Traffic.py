# -*- coding: utf-8 -*-
"""

TAIDE LLM for Traffic Sign

@date: 2024/03/30
@author: Prof. Ming-Hseng Tseng
"""
import os
os.chdir("C:/Users/user/OneDrive/桌面/image_ocr")

import tensorflow as tf
import pandas as pd
import numpy as np
import requests


### Load and Inspect Image Feature Set
ds_feature = pd.read_csv('C:/Users/user/OneDrive/桌面/extract_features_EfficientNetV2B2.csv')
filenames = ds_feature.iloc[:,0].values #['filename']
signs = ds_feature.iloc[:,1].values #['sign']
features = ds_feature.iloc[:,2::].values
print('Number of Total Images: ', len(filenames))
print('Number of Total Features: ', len(features[0]))

### Input Image Filename for Query
filename = input ( "輸入查詢交通號誌的影像路徑名稱: " )


### Extract Image Feature for Query
## preporcessing
image = tf.io.read_file(filename)
image = tf.image.decode_jpeg(image)
image = tf.image.resize(image, (224, 224))
#print(image.shape)
## extract feature
base_model = tf.keras.applications.EfficientNetV2B2(input_shape=(224,224,3), include_top=False, weights='imagenet')
inputs = tf.keras.Input(shape=(224,224,3))
x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
gap = tf.keras.layers.GlobalAveragePooling2D()(x)    
model = tf.keras.Model(inputs, gap)
image = (np.expand_dims(image,0))
#print(image.shape)
extract_features = model.predict(image)
#print(extract_features.shape)
img_feature = extract_features[0]


### Calculate Cosine_Similarity
cos_sim = np.zeros((len(filenames),))
for i in range(len(filenames)):
    X = img_feature
    Y = features[i]
    cos_sim[i] = np.dot(X,Y) / (np.linalg.norm(X)*np.linalg.norm(Y))
    #print(cos_sim[i])


### Find the Nearest Image Index
idx = np.argmax(cos_sim)
print("輸入影像最接近的交通號誌: ", signs[idx])


### Taide LLM API
host = ""
token = ""
headers = {
    "Authorization" : "" +token
}

#
query = signs[idx] #"讓路標誌" #"停車再開"
#chat
messages = [
{
"content" : " 你是一個來自台灣的AI助理，你的名字是 TAIDE，樂於以台灣人 的立場幫助使用者，會用繁體中文回答問題。 " ,
"role" : "system"
},
{
#"content" : "你剛剛參加了一場關於環保的公共演講，感受良多，希望能寫一封信給演 講者表示感謝。請根據你的感受和收穫，寫出一封感謝信的內容。" ,
"content" : "請解釋"+query+"這個交通號誌",
"role" : "user"
}
]
data = {
"model" : "TAIDE/a.2.0.0", #"TAIDE/b.11.0.0", #"TAIDE/a.1.0.0"
"messages" : messages,
"max_tokens" : 1000 ,
"temperature" : 0
}
r = requests.post(host+ "/chat/completions" , json=data, headers=headers)
res = r.json()[ "choices" ][ 0 ][ "message" ]
print (res)

