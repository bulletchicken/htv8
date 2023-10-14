import cv2 as cv
import numpy as np

import openai
import json

classes = ["background", "person", "bicycle", "car", "motorcycle",
  "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
  "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
  "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
  "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
  "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]


colors = np.random.uniform(0, 255, size=(len(classes), 3))
cam = cv.VideoCapture(0)

pb  = 'frozen_inference_graph.pb'
pbt = 'ssd_inception_v2_coco_2017_11_17.pbtxt'

cvNet = cv.dnn.readNetFromTensorflow(pb,pbt)   


humidity = 9
temperature = 8
light = 7
moisture = 3

clock=0

mood="happy"

messages = [ {"role": "user", "content": "You are plant that can speak and is given the personality of a dog"} ] 

response = ""

openai.api_key = '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'
# Ask a question and set the AI's role
def engageConversation(response):
    first = True

    while response!="bye":
        if first:
            if(response!=""):
            #initial conversation
                response = " holding " + response
                messages.append( {"role": "user", "content": "You are " + mood + " and your friend just walked in " + response + ". In 1 short sentence, no introductions, in a cute way them to help you if you are below 5 on any stats. Stats out of 10: humidity " + str(humidity) + ", temperature " + str(temperature) + ", light " + str(light) + ", moisture" + str(moisture)}, )
                first = False
        else:
            messages.append( {"role": "user", "content": response}, )
 
        chat = openai.ChatCompletion.create( 
            model="gpt-3.5-turbo", 
            messages=messages, 
            max_tokens=30
        ) 
        reply = chat.choices[0].message.content 
        print(reply)
        #get input again
        #save as response
        response=str(input())

   

while True:
  ret_val, img = cam.read()
  rows = img.shape[0]
  cols = img.shape[1]
  cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

  cvOut = cvNet.forward()

  seen =[]
  master = False

  for detection in cvOut[0,0,:,:]:

    score = float(detection[2])
    if score > 0.66:
        idx = int(detection[1])   # prediction class index. 
        if classes[idx]!="person":
            seen.append(classes[idx])
        if classes[idx] == "person":
            master = True
        
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            

        label = "{}: {:.2f}%".format(classes[idx],score * 100)
        y = top - 15 if top - 15 > 15 else top + 15
        cv.putText(img, label, (int(left), int(y)),cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

  cv.imshow('my webcam', img)
  if(master):

    clock+=1
    #send a promp to openAI here

    if(clock>20):
        print("I see you are holding ", seen)
        my_string = ', '.join(map(str, seen))
        if(mood=="thirsty"):
            if "bottle" in seen:
                print("I see you are holding a waterbottle. Is that for me? I'm pretty thirsty")
        else:
           engageConversation(my_string)
           #clock=0
           break #start engaging in conversation than CV
  else:
    clock=0
        
    
  if cv.waitKey(1) == 27: 
    break 
cam.release()
cv.destroyAllWindows()