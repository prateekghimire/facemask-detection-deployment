import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
model=tf.keras.models.load_model('trained.model')

st.write("""
        # FACE MASK PREDICTION
""")

st.write("""
            ## Developed by Pratik Ghimire
""")

file=st.file_uploader("Please upload image ", type=['jpg','png','jpeg'])

st.write('Note: Upload clear HD image only')

def import_and_predict(image_data,model):
    size=(100,100)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    image=np.asarray(image)
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    normal=img/255.0
    reshaped=np.reshape(normal,(1,100,100,1))
    prediction=model.predict(reshaped)
    result=np.argmax(prediction,axis=1)[0]
    
    return result 

if file is None:
    #st.text("Please upload an image ")
    pass
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)

    if prediction==1:
        st.write('''
        # The person is not wearing mask. Please buy a mask.''')
    elif prediction==0:
        st.write('''
       # The person is wearing mask. Good Job.
        ''')


