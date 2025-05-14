WELCOME TO SKINSAVERS!
------
This application enables you to scan your skin for skin cancer quickly on any device, which might save you thousands of dollars — or provide you with a few years of additional life. Simply upload at least one picture of your skin (uploading multiple is even better - as it allows for you to have a more complete analysis of your skin). The application will then quickly let you know whether skin cancer is detected. If it is, you’ll immediately receive valuable information, including the type of cancer, its estimated stage and progression, and clear next steps, such as effective treatment options and easy lifestyle adjustments.  
Python Documentation:  
- For the machine learning model we used the Ham10000 dataset to train a pretrained Densenet121
- For implementing the densenet121 model, we used this code as inspiration (https://www.kaggle.com/code/mathewkouch/ham10000-skin-lesion-classifier-82-pytorch)
- We reached an accuracy of 79% after 8 hours of training
JavaScript/Frontend Info:
- We used plain JavaScript (No Node.js or any extra frameworks)
- We used several modules: TensorFlow.JS, ONNX Web Runtime, Groq API/SDK (SDK almost always fails so it falls back to the API)
- Tensorflow.js: Uses the model file (.ONNX) and runs each image through the model
- ONNX Web Runtime: Allows us to use our ONNX model file (converted from PyTorch) 
- Groq Chatbot AI API/SDK: Groq is the chatbot API we are using as our AI assistant - giving world-class analysis anywhere in the world (this website works on all devices including computers and smartphones - basically any electronic device with a camera/file system and a web browser)
- We used complex HTML, CSS, and JS functions to build our mainline and side features.
