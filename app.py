from flask import Flask, render_template, redirect , request
import classifier
from classifier import model


#__name__==__main__
app=Flask(__name__)
@app.route('/',methods=['GET'])
def hello():
    return render_template("index.html")

@app.route('/',methods=['POST'])
def images():
    if request.method == 'POST':
        imagefile=request.files['userfile']
        image_path= "./static/{}".format(imagefile.filename)
        imagefile.save(image_path)

        prediction , probability= classifier.predict_image(image_path,model)

        result_dic= {
            'image' : image_path,
            'result' : prediction,
            'prob' : probability
            }

        

        
    return render_template("index.html", your_result = result_dic)




        

    # return render_template("index.html", your_result = result_dic)
if __name__== '__main__':
    app.run(port=2000, debug=True)