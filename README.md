The goal of this project is to create a GUI application that is capable of detecting objects (humans) in a selected image and adding more possible features to this application that might help National Intelligence to detect humans from an image with more accuracy than the human eye especially when there are multiple humans in an image. I created this project as my Minor Project-3 for my M.C.A. academic curriculum.
My journey of creating this project began by researching what options are available online and seeking GitHub repositories of people who already worked in the same domain. I learned about image annotation, and how to perform cleaning on a dataset of multiple images and their corresponding text files with the coordinates of the person in images in my dataset. Then I split my dataset into an 80:20 ratio of training and testing data. Luckily, I didn't have to annotate every single image because datasets for over 600 classes of objects are already available on the Google Open Images website with label files as well for each class. In the code part, first I designed a GUI application in Python using the Tkinter library and added the necessary components to my Tkinter window. While I was all set with the GUI, started to code for image transformation through the PIL library into the required format required for my model. I picked "Mask R-CNN (Regional Convolutional Neural Network) ResNet 50 FPN" as my object detection model. After the training, testing, and prediction of the model, I jump on adding more features to this project to make it more unique. Importantly, I added a feature of Age and Gender Estimation that can estimate the age, and gender of a person only if there is only a single human detected in the image through the DeepFace model of Facebook. After the coding and testing phase had ended well, I started working on my Documentation part and ended this project with the satisfaction of my Mentor. After documentation, I wrote a research paper on this project & published it in the International Journal of Computing & Artificial Intelligence.
I added various features to this project like Object (Human) Detection as the main core feature of this application. Along with object detection, the next key feature was Age and Gender Estimation and also the count of the detected humans in an image.
Moreover, I used Spyder IDE of Anaconda, and various python libraries (numpy, pandas, deepface, PIL, PyTorch, tkinter).

Research Paper published for this project link: https://www.computersciencejournals.com/ijcai/archives/2024.v5.i1.A.78

Following are few screenshots from this system:

![1](https://github.com/itsmesj/Object_Detection/assets/81063467/45b9d64b-beec-4e28-aeeb-6b141a33f0ad)
![2](https://github.com/itsmesj/Object_Detection/assets/81063467/9c9aee15-2ecf-4d8a-8e12-7cdefa2c8cc0)
![3](https://github.com/itsmesj/Object_Detection/assets/81063467/97fdee69-fee6-44d9-ab2c-df9d4147972c)
![4](https://github.com/itsmesj/Object_Detection/assets/81063467/dd0b500e-dd37-4ce4-b4d2-8856b06a7cb5)
![5](https://github.com/itsmesj/Object_Detection/assets/81063467/10e0909d-e026-4e7b-a357-97f885220f42)
![6](https://github.com/itsmesj/Object_Detection/assets/81063467/cf787f57-f2db-4a64-9fc8-020dcc4b160c)
By setting the threshold value according to customized requirements, the system detects the confidence threshold setted between 0 and 1.
![7](https://github.com/itsmesj/Object_Detection/assets/81063467/1c96e3fc-3beb-4117-8f18-ee8eb46f8722)
![8](https://github.com/itsmesj/Object_Detection/assets/81063467/40a8e5c0-8ffc-4f05-9256-8ad300bac1e4)
![9](https://github.com/itsmesj/Object_Detection/assets/81063467/6aa67500-10cd-42e3-89f3-00a0ac694f0a)
![10](https://github.com/itsmesj/Object_Detection/assets/81063467/48a3bf59-7597-421f-b33e-140006e7d927)
![11](https://github.com/itsmesj/Object_Detection/assets/81063467/dbb72889-9493-40d6-b2d6-667d15d6776c)
![12](https://github.com/itsmesj/Object_Detection/assets/81063467/312c4a84-4bdb-41a8-b02f-cbf0aae28f64)
![13](https://github.com/itsmesj/Object_Detection/assets/81063467/79f71c4b-29c5-4688-ab67-d2e0806ecccf)
![14](https://github.com/itsmesj/Object_Detection/assets/81063467/ee0cd451-a747-4dc1-8de5-91a924b2948b)
![15](https://github.com/itsmesj/Object_Detection/assets/81063467/57a7eeac-b8af-4078-bb42-34a8b6659037)
![16](https://github.com/itsmesj/Object_Detection/assets/81063467/91de19b8-f07e-4abe-885d-e2fea8104fac)
![17](https://github.com/itsmesj/Object_Detection/assets/81063467/c92fce2b-1dc8-4bd1-8bec-84b0de090f7e)
![18](https://github.com/itsmesj/Object_Detection/assets/81063467/c17eab70-0bf4-4054-9827-4cb2ea7b106d)
![19](https://github.com/itsmesj/Object_Detection/assets/81063467/c97d2be9-6d69-4479-804b-7f0c8c9a79ca)
![20](https://github.com/itsmesj/Object_Detection/assets/81063467/4b25bdd8-cf86-4b18-a483-ed6ee07dca57)
![21](https://github.com/itsmesj/Object_Detection/assets/81063467/a295c367-51dd-419c-b1ef-2bbbf59b0001)
