import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

classifier = load_model('detect_emo_model.h5')

detector = cv2.CascadeClassifier('frontalface.xml')

class_labels = ['Marah','Senang','Netral','Sedih']

class GUI(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        w,h = 500, 400
        master.minsize(width=w, height=h)
        master.maxsize(width=w, height=h)
        self.pack()

        self.file = tk.Button(self, text='Browse', command=self.choose)
        self.choose = tk.Label(self, text="Choose file").pack()
        self.imageFile = Image.open("pic.png")
        self.image = ImageTk.PhotoImage(self.imageFile)
        self.label = tk.Label(image=self.image)
        fontStyle = tkFont.Font(family="Lucida Grande", size=20)
        self.emotion = tk.Label(self, text="Detected emotion", font=fontStyle)
        
        self.file.pack()
        self.label.pack()
        self.emotion.pack(side = tk.BOTTOM)

    def choose(self):
        self.filename = filedialog.askopenfilename(initialdir = "/",
                                              title = "Select a File",
                                              filetypes = [("Images",".jpg .jpeg .png")]
                                              )
        imageFile = Image.open(self.filename)
        wpercent = (300/float(imageFile.size[0]))
        hsize = int((float(imageFile.size[1])*float(wpercent)))
        imageFile = imageFile.resize((300,hsize), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(imageFile)
        self.label.configure(image=self.image)
        self.label.image=self.image
        self.predict()

    def predict(self):
        converted = cv2.imread(self.filename)
        gray = cv2.cvtColor(converted,cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray,1.3,5)

        if len(faces) != 0:
            for (x,y,w,h) in faces:
                cv2.rectangle(converted,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)
                    preds = classifier.predict(roi)[0]
                    self.emotion.configure(text=class_labels[preds.argmax()])
                else:
                    self.emotion.configure(text='No face detected')
        else:
            roi_gray = cv2.resize(gray,(48,48),interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                preds = classifier.predict(roi)[0]
                print(preds)
                self.emotion.configure(text=class_labels[preds.argmax()])
            else:
                self.emotion.configure(text='No face detected')


root = tk.Tk()
root.title('Emotion detector')
app = GUI(master=root)
app.mainloop()
root.destroy()