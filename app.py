import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle#we need it bcz we need to read our pickle file that we have created


#first create flask app
app = Flask(__name__)

#read our pickle file that we have created
model = pickle.load(open('model.pkl', 'rb'))





#below is basically our home page,by default root page ('/') which is just-like  slash will actually render a template called
#return render_template('index.html')
#@app.route('/')
#def home():
#    return render_template('index.html')
#then this will run from index.html
#<input type="text" name="experience" placeholder="Experience" required="required" />
#        <input type="text" name="test_score" placeholder="Test Score" required="required" />
#		<input type="text" name="interview_score" placeholder="Interview Score" required="required" />






@app.route('/')
def home():
    return render_template('index.html')








#but I have also created '/predict' which is a POST Method,where I am providing some features to my model.pkl file
#so that my model will take some inputs and give us some outputs
#or we can say this func  (def predict():)   is like a web api
#this func takes values rfom all fields and basically request lib tekes those values from text fields and store it in int_feat
##after this convert it into array
#and then we predict
#after it,we just found output
#after it we again render index.html,but now I will some data which is prediction_text
#this prediction_text will get replace over here {{ prediction_text }}        (from index.html)    

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))





#finally we have a main func which will run this entire flask
if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    
#now to run app.py
#first open anaconda command prompt,first change ur drive to that drive where we have our coding file ie app.py
#c:\users\mcr\F:

#|mention path now 

#now run your file by  ___________________________________________________  python app.py    
 #now we get local IP on which we cam do prediction http://127.0.0.1:5000/ open this link on chrome or any other browser 
#but this is a local IP and it is not publically avalaible  
    
#notw when we click on predict button,it willl simply run predict function of app.py
   
    
 #now using heroku for publically available   
    