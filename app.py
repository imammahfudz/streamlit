import time
import numpy as np
import streamlit as st
import urllib.request
import tomli
from PIL import Image
from utils import *
from keras.utils import load_img, img_to_array  


with open("config.toml", mode="rb") as fp:
  config = tomli.load(fp)

pc = st.get_option('theme.primaryColor')
bc = st.get_option('theme.backgroundColor')
sbc = st.get_option('theme.secondaryBackgroundColor')
tc = st.get_option('theme.textColor')

labels = gen_labels()

html_temp = '''
    <div style =  padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Klasifikasi Batuan</h1></center>
    
    </div>
    '''

st.markdown(html_temp, unsafe_allow_html=True)
vert_space = '<div style="padding: 180px 5px;"></div>'
html_temp = '''
    <div>
    <h2></h2>
    <center><h3>Silakan unggah citra untuk menemukan kategorinya</h3></center>
    </div>
    '''
st.set_option('deprecation.showfileUploaderEncoding', False)
st.markdown(html_temp, unsafe_allow_html=True)
opt = st.selectbox("Pilih metode untuk klasifikasi citra\n", ('Silakan Pilih', 'Unggah citra via link', 'Unggah citra dari piranti'))
if opt == 'Unggah citra dari piranti':
    file = st.file_uploader('Pilih', type = ['jpg', 'png', 'jpeg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        image = Image.open(file).convert("RGB")

elif opt == 'Unggah citra via link':

  try:
    img = st.text_input('Masukkan URL Gambar...')
    image = Image.open(urllib.request.urlopen(img)).convert("RGB")
    
  except:
    if st.button('Submit'):
      show = st.error("Masukkan URL citra yang Benar")
      time.sleep(4)
      show.empty()

try:
  # st.markdown(html_temp, vert_space, unsafe_allow_html=True)
  
  col1, col2, col3 = st.columns(3)
  with col1:
    pass
  # with col2:
  #   pass
  with col2:
    st.image(image, use_column_width=False, width = 200, caption = 'Uploaded Image')
    if st.button("ㅤㅤㅤㅤPredictㅤㅤㅤㅤ"):
      img = preprocess(image)
      model = CNN_Model()
      model.load_weights("model.h5")
      prediction = model.predict(img[np.newaxis, ...])
      proba = np.max(prediction[0], axis=-1)
      print("Probabilitas :", np.max(prediction[0], axis=-1))
      st.info('ㅤㅤBatuan {} '.format(labels[np.argmax(prediction[0], axis=-1)]))
      st.info('ㅤㅤAkurasi = '+ str(proba))
  with col3:
    pass
  # with col5:
  #   pass
except Exception as e:
  st.info(e)
  pass
 
  # if image is not None:
    # st.image(image, width = 300, caption = 'Uploaded Image')
    # if st.button('aaaaaa'):
    #     img = preprocess(image)
    #     model = CNN_Model()
    #     model.load_weights("model.h5")
    #     prediction = model.predict(img[np.newaxis, ...])
    #     proba = np.max(prediction[0], axis=-1)
    #     print("Probabilitas :", np.max(prediction[0], axis=-1))
    #     st.info('Citra diklasifikasikan sebagai = {} '.format(labels[np.argmax(prediction[0], axis=-1)]))
    #     st.info('Akurasi = '+ str(proba))
# except Exception as e:
#   st.info(e)
#   pass
