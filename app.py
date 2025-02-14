from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
 
app = Flask(__name__)




verbose_name = {


0: 'No_DR',
1: 'Mild',
2: 'Moderate',
3: 'Severe',
4: 'Proliferate_DR'


}
 
DenseNet = load_model('DenseNet201.h5')
mobilenet = load_model('MobileNetV2.h5')

# def predict_label(img_path):
# 	test_image = image.load_img(img_path, target_size=(150,150))
# 	test_image = image.img_to_array(test_image)/255.0
# 	test_image = test_image.reshape(-1, 1, 150, 150, 3)

# 	predict_x=lstm.predict(test_image) 
# 	classes_x=np.argmax(predict_x,axis=1)
	
# 	return verbose_name [classes_x[0]]

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(224,224))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 224,224,3)

	predict_x=mobilenet.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verbose_name [classes_x[0]] 

def denseNet(img_path):
	test_image = image.load_img(img_path, target_size=(224,224))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 224,224,3)

	predict_x=DenseNet.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verbose_name [classes_x[0]] 

@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')   
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


# @app.route("/submit", methods = ['GET', 'POST'])
# def get_output():
# 	if request.method == 'POST':
# 		img = request.files['my_image']

# 		img_path = "static/tests/" + img.filename	
# 		img.save(img_path)
# 		#plt.imshow(img)
# 		predict_result = predict_label(img_path)
# 		# predict_result = predict_label(img_path)
		 

# 		#print(predict_result)
# 	return render_template("prediction.html", prediction = predict_result, img_path = img_path)

@app.route("/submit", methods = ['GET', 'POST'])
def submit():
	predict_result = None
	img_path = None
	model = None
	if request.method == 'POST':
		img = request.files['my_image']
		model = request.form['model']
		print(model)
	    # predict_result = "Prediction: Success" 
		img_path = "static/tests/" + img.filename	
		img.save(img_path)
		#plt.imshow(img)

		if model == 'DenseNet201':

		     predict_result = denseNet(img_path)
		elif model == 'MobileNetV2':
			 predict_result = predict_label(img_path)
		 
		 	  
	return render_template("prediction.html", prediction = predict_result, img_path = img_path, model = model)

@app.route("/chart")
def chart():
	return render_template('chart.html') 
@app.route("/performance")
def performance():
	return render_template('performance.html')  	


	

	
if __name__ =='__main__':
	app.run(debug = True)


	

	


