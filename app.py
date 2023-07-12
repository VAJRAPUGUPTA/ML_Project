from flask import Flask,render_template,Response
from flask_wtf import FlaskForm
from wtforms import FileField , SubmitField
from werkzeug.utils import secure_filename
import os 
from wtforms.validators import InputRequired
import cv2 
import numpy as np
import imutils
from tensorflow.keras.models import load_model
from datetime import date
import math


app =  Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
camera = cv2.VideoCapture(0)

weights0_path = './input/detect-person-on-motorbike-or-scooter/yolov3-obj_final.weights'
configuration0_path = './input/detect-person-on-motorbike-or-scooter/yolov3_pb.cfg'

probability_minimum = 0.5
threshold = 0.3

COLORS = [(0,255,0),(0,0,255)]

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

network0 = cv2.dnn.readNetFromDarknet(configuration0_path, weights0_path)
layers_names0_all = network0.getLayerNames()
layers_names0_output = [layers_names0_all[i-1] for i in network0.getUnconnectedOutLayers()]
labels0 = open('./input/detect-person-on-motorbike-or-scooter/coco.names').read().strip().split('\n')

model = load_model('helmet-nonhelmet_cnn.h5')
print('model loaded!!!')


layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()


# for helmet
weights1_path = './input/helmet-detection-yolov3/yolov3-helmet.weights'
configuration1_path = './input/helmet-detection-yolov3/yolov3-helmet.cfg'


network1 = cv2.dnn.readNetFromDarknet(configuration1_path, weights1_path)
layers_names1_all = network1.getLayerNames()
layers_names1_output = [layers_names1_all[i-1] for i in network1.getUnconnectedOutLayers()]
labels1 = open('./input/helmet-detection-yolov3/helmet.names').read().strip().split('\n')


np.random.seed(42)
colours0 = np.random.randint(0,255,size=(len(labels0),3),dtype='uint8')
colours1 = np.random.randint(0,255,size=(len(labels1),3),dtype='uint8')

def helmet(imagee,l1):
    tempp = imagee
    blob1 = cv2.dnn.blobFromImage(tempp,1/255.0,(416,416),swapRB=True,crop=False)
    blob_to_show = blob1[0,:,:,:].transpose(1,2,0)
    network1.setInput(blob1)
    output_from_network1 = network1.forward(output_layers)
    temp = tempp
    height,width = tempp.shape[:2]
    cc = 0
    for result in output_from_network1:
        for detection in result:
            scores = detection[5:]
            class_current=np.argmax(scores)
            confidence_current=scores[class_current]
            if confidence_current>probability_minimum:
                box_current=detection[0:4]*np.array([width,height,width,height])
                x_center,y_center,box_width,box_height=box_current.astype('int')
                x_min=int(x_center-(box_width/2))
                y_min=int(y_center-(box_height/2))
                color = [int(c) for c in COLORS[class_current]]
                cv2.rectangle(temp, (x_min, y_min), (x_min + box_width, y_min + box_height), color, 7)
                cc = cc+1

    if cc>=1:
        return 1
    else:
        return 0



def check(imagee,l1,directory):
                tempp = imagee
                blob1 = cv2.dnn.blobFromImage(tempp,1/255.0,(416,416),swapRB=True,crop=False)
                blob_to_show = blob1[0,:,:,:].transpose(1,2,0)
                net.setInput(blob1)
                outs = net.forward(output_layers)
                temp = tempp
                height,width = tempp.shape[:2]
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.4:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            color = [int(c) for c in COLORS[class_id]]
                            if class_id==0:
                                helmet_roi = temp[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
                            else:
                                x_h = x-60
                                y_h = y-350
                                w_h = w+100
                                h_h = h+100
                                cv2.rectangle(temp, (x, y), (x + w, y + h), color, 2)
                                
class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Upload File")


def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)
    count  = 0
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            img = imutils.resize(frame,height=500)
            parent = os.getcwd().replace("\\","/") + '/static/numberplates'
            path = parent+'/'+'livestream'
            image_names = os.listdir(path)
            li = len(image_names)
            COLORS = [(0,255,0),(0,0,255)]
            layer_names = net.getLayerNames()
            output_layers = net.getUnconnectedOutLayersNames()
            height, width = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            confidences = []
            boxes = []
            classIds = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        classIds.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h = boxes[i]
                    color = [int(c) for c in COLORS[classIds[i]]]
                    if classIds[i]==0:
                        helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
                    else:
                        x_h = x-60
                        y_h = y-350
                        w_h = w+100
                        h_h = h+100
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                        if y_h>0 and x_h>0:
                            h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
                            c = helmet_or_nohelmet(h_r)
                            cv2.putText(img,['helmet','no-helmet'][c],(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)                
                            cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 10)
                            if c==1:
                                imagg = img[y:y+h,x:x+w]
                                today = date.today()
                                today = str(today)
                                path = path+'/'+today
                                if not os.path.exists(path):
                                    os.mkdir(path)
                                cv2.imwrite('{}_{}.{}'.format(os.path.join(path,'img'), str(li), 'jpg'), imagg)


            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ret,buffer=cv2.imencode('.jpg',img)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def helmet_or_nohelmet(helmet_roi):
	try:
		helmet_roi = cv2.resize(helmet_roi, (224, 224))
		helmet_roi = np.array(helmet_roi,dtype='float32')
		helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
		helmet_roi = helmet_roi/255.0
		return int(model.predict(helmet_roi)[0][0])
	except:
			pass

@app.route('/',methods = ["GET","POST"])
@app.route('/home',methods = ["GET","POST"])
def home():
    form = UploadFileForm()
    global camera
    camera.release()
    if form.validate_on_submit():
        file = form.file.data
        directory = file.filename
        seperator = '.'
        directory = directory.split(seperator,1)[0]
        parent = os.getcwd().replace("\\","/") + '/static/numberplates'
        path = parent+'/'+directory
        if not os.path.exists(path):
            os.mkdir(path)
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))

        cap = cv2.VideoCapture(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))

        COLORS = [(0,255,0),(0,0,255)]
        xii1 = 0

        layer_names = net.getLayerNames()
        output_layers = net.getUnconnectedOutLayersNames()
        

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter('output.avi', fourcc, 5,(888,500))

        ret = True
        xii = 1

        while ret:

            ret, img = cap.read()
            if img is None or xii1==200:
                break
            img = imutils.resize(img,height=500)
            height, width = img.shape[:2]

            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)

            confidences = []
            boxes = []
            classIds = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)

                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        classIds.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h = boxes[i]
                    color = [int(c) for c in COLORS[classIds[i]]]
                    if classIds[i]==0: #bike
                        helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
                    else: #number plate
                        x_h = x-60
                        y_h = y-350
                        w_h = w+100
                        h_h = h+100
                        if y_h>0 and x_h>0:
                            h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
                            c = helmet_or_nohelmet(h_r)
                            if c==1:
                                imagg = img[y:y+h,x:x+w]
                                cv2.imwrite('{}_{}.{}'.format(os.path.join(path,'img'), str(xii), 'jpg'), imagg)
                                xii = xii + 1


            writer.write(img)
            xii1 = xii1+1

            if cv2.waitKey(1) == 27:
                break

        writer.release()
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        image_names = os.listdir(path)
        for i in range(len(image_names)):
            image_names[i] = '../static/numberplates/'+ directory+'/'+image_names[i]
        return render_template('images.html',image_name = image_names)
    return render_template('index.html',form = form)




@app.route('/image_upload',methods = ["GET","POST"])
def image_up():
    form = UploadFileForm()
    global camera
    camera.release()
    ans = ""
    if form.validate_on_submit():
        file = form.file.data
        directory = file.filename
        seperator = '.'
        directory = directory.split(seperator,1)[0]
        parent = os.getcwd().replace("\\","/") + '/static/numberplates'
        path = parent+'/'+directory
        if not os.path.exists(path):
            os.mkdir(path)
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        pa  = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        img = cv2.imread(pa)

        image_input = img
        blob = cv2.dnn.blobFromImage(image_input,1/255.0,(416,416),swapRB=True,crop=False)
        blob_to_show = blob[0,:,:,:].transpose(1,2,0)
        network0.setInput(blob)
        output_from_network0 = network0.forward(layers_names0_output)

        h,w = image_input.shape[:2]
        height,width = image_input.shape[:2]

        l1 = 0
        for result in output_from_network0:
            for detection in result:
                scores = detection[5:]
                class_current=np.argmax(scores)
                confidence_current=scores[class_current]
                l1=scores[class_current]
                if confidence_current>=probability_minimum:
                    box_current=detection[0:4]*np.array([w,h,w,h])
                    x_center,y_center,box_width,box_height=box_current.astype('int')
                    x_min=int(math.ceil(x_center-(box_width/2)))
                    y_min=int(math.ceil(y_center-(box_height/2)))
                    imagee = image_input[y_min:y_min+int(box_height),x_min:x_min+int(box_width)]
                    if helmet(imagee,l1)==0:
                        check(imagee,l1,directory)
                        pa1 = './static/numberplates/'+str(directory)+'/img_box'+str(l1)+'.jpg'
                        l1 = l1 + 1
                        cv2.imwrite(pa1, imagee)

        image_names = os.listdir('./static/numberplates/'+directory+'/')
        for i in range(len(image_names)):
            image_names[i] = '../static/numberplates/'+ directory+'/'+image_names[i]
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
        return render_template('images.html',image_name = image_names)
    return render_template('image_upload.html',form = form)


@app.route('/stream',methods = ["GET","POST"])
def video_page():
    return render_template('video.html')


@app.route('/video',methods = ["GET","POST"])
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)


app =  Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
camera = cv2.VideoCapture(0)

weights0_path = './input/detect-person-on-motorbike-or-scooter/yolov3-obj_final.weights'
configuration0_path = './input/detect-person-on-motorbike-or-scooter/yolov3_pb.cfg'

probability_minimum = 0.5
threshold = 0.3

COLORS = [(0,255,0),(0,0,255)]

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

network0 = cv2.dnn.readNetFromDarknet(configuration0_path, weights0_path)
layers_names0_all = network0.getLayerNames()
layers_names0_output = [layers_names0_all[i-1] for i in network0.getUnconnectedOutLayers()]
labels0 = open('./input/detect-person-on-motorbike-or-scooter/coco.names').read().strip().split('\n')

model = load_model('helmet-nonhelmet_cnn.h5')
print('model loaded!!!')


layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()


# for helmet
weights1_path = './input/helmet-detection-yolov3/yolov3-helmet.weights'
configuration1_path = './input/helmet-detection-yolov3/yolov3-helmet.cfg'


network1 = cv2.dnn.readNetFromDarknet(configuration1_path, weights1_path)
layers_names1_all = network1.getLayerNames()
layers_names1_output = [layers_names1_all[i-1] for i in network1.getUnconnectedOutLayers()]
labels1 = open('./input/helmet-detection-yolov3/helmet.names').read().strip().split('\n')


np.random.seed(42)
colours0 = np.random.randint(0,255,size=(len(labels0),3),dtype='uint8')
colours1 = np.random.randint(0,255,size=(len(labels1),3),dtype='uint8')

def helmet(imagee,l1):
    tempp = imagee
    blob1 = cv2.dnn.blobFromImage(tempp,1/255.0,(416,416),swapRB=True,crop=False)
    blob_to_show = blob1[0,:,:,:].transpose(1,2,0)
    network1.setInput(blob1)
    output_from_network1 = network1.forward(output_layers)
    temp = tempp
    height,width = tempp.shape[:2]
    cc = 0
    for result in output_from_network1:
        for detection in result:
            scores = detection[5:]
            class_current=np.argmax(scores)
            confidence_current=scores[class_current]
            if confidence_current>probability_minimum:
                box_current=detection[0:4]*np.array([width,height,width,height])
                x_center,y_center,box_width,box_height=box_current.astype('int')
                x_min=int(x_center-(box_width/2))
                y_min=int(y_center-(box_height/2))
                color = [int(c) for c in COLORS[class_current]]
                cv2.rectangle(temp, (x_min, y_min), (x_min + box_width, y_min + box_height), color, 7)
                cc = cc+1

    if cc>=1:
        return 1
    else:
        return 0



def check(imagee,l1,directory):
                tempp = imagee
                blob1 = cv2.dnn.blobFromImage(tempp,1/255.0,(416,416),swapRB=True,crop=False)
                blob_to_show = blob1[0,:,:,:].transpose(1,2,0)
                net.setInput(blob1)
                outs = net.forward(output_layers)
                temp = tempp
                height,width = tempp.shape[:2]
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.4:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            color = [int(c) for c in COLORS[class_id]]
                            if class_id==0:
                                helmet_roi = temp[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
                            else:
                                x_h = x-60
                                y_h = y-350
                                w_h = w+100
                                h_h = h+100
                                cv2.rectangle(temp, (x, y), (x + w, y + h), color, 2)
                                
class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Upload File")


def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)
    count  = 0
    while True:
        success,frame=camera.read()
        if not success:
            break
        else:
            img = imutils.resize(frame,height=500)
            parent = os.getcwd().replace("\\","/") + '/static/numberplates'
            path = parent+'/'+'livestream'
            image_names = os.listdir(path)
            li = len(image_names)
            COLORS = [(0,255,0),(0,0,255)]
            layer_names = net.getLayerNames()
            output_layers = net.getUnconnectedOutLayersNames()
            height, width = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            confidences = []
            boxes = []
            classIds = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        classIds.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h = boxes[i]
                    color = [int(c) for c in COLORS[classIds[i]]]
                    if classIds[i]==0:
                        helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
                    else:
                        x_h = x-60
                        y_h = y-350
                        w_h = w+100
                        h_h = h+100
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 7)
                        if y_h>0 and x_h>0:
                            h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
                            c = helmet_or_nohelmet(h_r)
                            cv2.putText(img,['helmet','no-helmet'][c],(x,y-100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)                
                            cv2.rectangle(img, (x_h, y_h), (x_h + w_h, y_h + h_h),(255,0,0), 10)
                            if c==1:
                                imagg = img[y:y+h,x:x+w]
                                today = date.today()
                                today = str(today)
                                path = path+'/'+today
                                if not os.path.exists(path):
                                    os.mkdir(path)
                                cv2.imwrite('{}_{}.{}'.format(os.path.join(path,'img'), str(li), 'jpg'), imagg)


            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ret,buffer=cv2.imencode('.jpg',img)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def helmet_or_nohelmet(helmet_roi):
	try:
		helmet_roi = cv2.resize(helmet_roi, (224, 224))
		helmet_roi = np.array(helmet_roi,dtype='float32')
		helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
		helmet_roi = helmet_roi/255.0
		return int(model.predict(helmet_roi)[0][0])
	except:
			pass

@app.route('/',methods = ["GET","POST"])
@app.route('/home',methods = ["GET","POST"])
def home():
    form = UploadFileForm()
    global camera
    camera.release()
    if form.validate_on_submit():
        file = form.file.data
        directory = file.filename
        seperator = '.'
        directory = directory.split(seperator,1)[0]
        parent = os.getcwd().replace("\\","/") + '/static/numberplates'
        path = parent+'/'+directory
        if not os.path.exists(path):
            os.mkdir(path)
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))

        cap = cv2.VideoCapture(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))

        COLORS = [(0,255,0),(0,0,255)]
        xii1 = 0

        layer_names = net.getLayerNames()
        output_layers = net.getUnconnectedOutLayersNames()
        

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter('output.avi', fourcc, 5,(888,500))

        ret = True
        xii = 1

        while ret:

            ret, img = cap.read()
            if img is None or xii1==200:
                break
            img = imutils.resize(img,height=500)
            height, width = img.shape[:2]

            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layers)

            confidences = []
            boxes = []
            classIds = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)

                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        classIds.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h = boxes[i]
                    color = [int(c) for c in COLORS[classIds[i]]]
                    if classIds[i]==0: #bike
                        helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
                    else: #number plate
                        x_h = x-60
                        y_h = y-350
                        w_h = w+100
                        h_h = h+100
                        if y_h>0 and x_h>0:
                            h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
                            c = helmet_or_nohelmet(h_r)
                            if c==1:
                                imagg = img[y:y+h,x:x+w]
                                cv2.imwrite('{}_{}.{}'.format(os.path.join(path,'img'), str(xii), 'jpg'), imagg)
                                xii = xii + 1


            writer.write(img)
            xii1 = xii1+1

            if cv2.waitKey(1) == 27:
                break

        writer.release()
        cap.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        image_names = os.listdir(path)
        for i in range(len(image_names)):
            image_names[i] = '../static/numberplates/'+ directory+'/'+image_names[i]
        return render_template('images.html',image_name = image_names)
    return render_template('index.html',form = form)




@app.route('/image_upload',methods = ["GET","POST"])
def image_up():
    form = UploadFileForm()
    global camera
    camera.release()
    ans = ""
    if form.validate_on_submit():
        file = form.file.data
        directory = file.filename
        seperator = '.'
        directory = directory.split(seperator,1)[0]
        parent = os.getcwd().replace("\\","/") + '/static/numberplates'
        path = parent+'/'+directory
        if not os.path.exists(path):
            os.mkdir(path)
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        pa  = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        img = cv2.imread(pa)

        image_input = img
        blob = cv2.dnn.blobFromImage(image_input,1/255.0,(416,416),swapRB=True,crop=False)
        blob_to_show = blob[0,:,:,:].transpose(1,2,0)
        network0.setInput(blob)
        output_from_network0 = network0.forward(layers_names0_output)

        h,w = image_input.shape[:2]
        height,width = image_input.shape[:2]

        l1 = 0
        for result in output_from_network0:
            for detection in result:
                scores = detection[5:]
                class_current=np.argmax(scores)
                confidence_current=scores[class_current]
                l1=scores[class_current]
                if confidence_current>=probability_minimum:
                    box_current=detection[0:4]*np.array([w,h,w,h])
                    x_center,y_center,box_width,box_height=box_current.astype('int')
                    x_min=int(math.ceil(x_center-(box_width/2)))
                    y_min=int(math.ceil(y_center-(box_height/2)))
                    imagee = image_input[y_min:y_min+int(box_height),x_min:x_min+int(box_width)]
                    if helmet(imagee,l1)==0:
                        check(imagee,l1,directory)
                        pa1 = './static/numberplates/'+str(directory)+'/img_box'+str(l1)+'.jpg'
                        l1 = l1 + 1
                        cv2.imwrite(pa1, imagee)

        image_names = os.listdir('./static/numberplates/'+directory+'/')
        for i in range(len(image_names)):
            image_names[i] = '../static/numberplates/'+ directory+'/'+image_names[i]
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
        return render_template('images.html',image_name = image_names)
    return render_template('image_upload.html',form = form)


@app.route('/stream',methods = ["GET","POST"])
def video_page():
    return render_template('video.html')


@app.route('/video',methods = ["GET","POST"])
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace;boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)