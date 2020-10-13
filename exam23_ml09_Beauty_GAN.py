#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches #Matplotlib 에서 이미지 나 일반 도형에 사각형을 그려야 할 때
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import numpy as np 


# In[2]:


detector = dlib.get_frontal_face_detector() #얼굴 찾아주는 detector ->사진에서 얼굴 영역만 찾아줌 
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat') #눈,코, 입 찾아주는 모델 


# In[3]:


img = dlib.load_rgb_image('./imgs/12.jpg') #이미지 불러오기
plt.figure(figsize=(16,10))
plt.imshow(img) #이미지 -> imshow


# In[4]:


img_result = img.copy() #이미지 복사
dets = detector(img) #얼굴 영역 찾아줌 
if len(dets) ==0: #이미지에 얼굴이 없는 경우 
    print('cannot find faces!')
else: 
    fig, ax = plt.subplots(1,figsize=(16,10)) #얼굴 이미지 
    for det in dets: #dets에는 얼굴의 영역들이 들어있음(왼쪽 맨 위가 기준점에서부터 맨 왼쪽(x), 위쪽(y) 높이 넓이)
        x, y, w, h = det.left(), det.top(), det.width(), det.height() 
        rect = patches.Rectangle((x, y),w,h,linewidth=2, edgecolor='r', 
                                 facecolor='none')
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()


# In[5]:


fig, ax = plt.subplots(1, figsize=(16,10)) #사진 1줄 fig : 도화지 
objs = dlib.full_object_detections() #포인트 찾아주는 object 객체 
for detection in dets: 
    s= sp(img, detection) #sp: shape point(얼굴 랜드마크 찾아줌) / 원본 이미지/ 선 -> 얼굴 영역 찾은 이미지 
    objs.append(s)
    for point in s.parts(): #랜드마크 parts : 5개 
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r') #포인트 설정 
        ax.add_patch(circle)
ax.imshow(img_result)


# In[6]:


faces = dlib.get_face_chips(img, objs, size=256, padding=0.3) #get_face_chips: 잘라줌 /objs를 넣어서 얼굴 영역만 자름
#padding : 얼굴 옆 여백 -> 여유공간 
fig, axes = plt.subplots(1, len(faces)+1, figsize = (20,16)) #1줄, subplot의 갯수 ->얼굴 4개 
axes[0].imshow(img) #원본이미지 처음으로 출력
for i, face in enumerate(faces):
    axes[i+1].imshow(face)


# In[7]:


#얼굴 찾기 -> 랜드마크 찾기 -> 얼굴 영역 자르기 

def align_faces(img):
    dets = detector(img)  #얼굴 찾기 dets: 얼굴들이 들어가있음  
    objs = dlib.full_object_detections() #objs는 비어있음. 포인트 찾아주는 object 객체 
    for detection in dets: #얼굴들 이미지 갯수만큼 for문 돔 
        s = sp(img, detection) # s: landmark 다섯개 점에 대한 정보 
        objs.append(s) #objs에 점 5개 저장 
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35) #faces 자른 얼굴 영역저장 --> 얼굴 이미지로 저장
    return faces #얼굴 이미지들

test_img = dlib.load_rgb_image('./imgs/03.jpg') #이미지 불러오기 
test_faces = align_faces(test_img) #함수에 테스트 이미지를 넣음  faces = test_faces
fig, axes = plt.subplots(1, len(test_faces)+1, figsize =(20,16)) #1줄에 얼굴이 4개면 5개를 그림(원본, 얼굴4개)
axes[0].imshow(test_img)
for i, face in enumerate(test_faces): #얼굴 이미지만 들어있음 
    axes[i+1].imshow(face)


# In[8]:


#https://github.com/Honlan/BeautyGAN/commit/911d2090784ff90006e988850f2f3e82c2320fc1
sess = tf.Session()#tensorflow는 session을 만들어서 fit
sess.run(tf.global_variables_initializer())#모델 초기화 

saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')


# In[9]:


def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2


# In[20]:


img1 = dlib.load_rgb_image('./imgs/12.jpg') #소스 이미지 
img1_faces = align_faces(img1)

img2 = dlib.load_rgb_image('./imgs/makeup/vFG56.png') #레퍼런스 이미지(메이크업)
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1,2,figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()


# In[21]:


src_img = img1_faces[0] #소스 이미지
ref_img = img2_faces[0] #레퍼런스 이미지 

X_img = preprocess(src_img) #모델에 넣어줄 이미지 -> 스케일링 필요
X_img = np.expand_dims(X_img, axis = 0)

Y_img = preprocess(ref_img) 
Y_img = np.expand_dims(Y_img, axis = 0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img}) #화장한 이미지 생성 해서 output 변수에 저장 
output_img = deprocess(output[0]) #다시 리스케일링해서 출력 

fig, axes = plt.subplots(1, 3, figsize=(20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()

#학습된 이미지로 했을 때 자연스러운 결과 


# In[ ]:




