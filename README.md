# Eigenfaces

**Group**: **ASA** _Anastasiia Livochka_, _Sofia Kholod_, _Andriy Dmytruk_  
**Goal**: Use the **Eigenfaces** approach to recognize our groupmates  

## Usage

##### Server
To run the local server with face recognition, you need to start:
```shell
python main.py server [FOLDER] [DATASET_FOLDER]
```
This will start the server at [localhost:3000](http://localhost:3000/). The parameter
folder is not required and needs to have the path to a dataset of normalized people
photos.  
This will train the algorithm on dataset and will be ready to recognize.  

Website supports taking photos online, uploading photos stored locally and switching camera.  
Website can also be used to **collect the dataset**. The second parameter `[DATASET_FOLDER]` is the folder 
where all dataset photos will be saved. The endpoint
[localhost:3000/eigenfaces/save](http://localhost:3000/eigenfaces/save) helps
you easily take the photos and automatically save them to dataset.

##### Dataset & Normalization
The original dataset can be found in at [Google Drive](https://drive.google.com/drive/folders/1Mr106DGi1xcxc7XxvzuWOh6SjTp76bv_?usp=sharing).  
The normalized dataset is stored in [normalized_apps](./normalized_apps) folder. So, if you just want to test
the recognition, you do not need to manually normalize the dataset.

Before running recognition, you need to normalize the photos in dataset.  
To do this run:   
```shell
python normalize_images.py [SOURCE_FOLDER] [RESULT_FOLDER]
```
The default parameters are `./apps_faces` and `./normalized_apps`  

##### Testing
Testing is performed automatically after normalization to inform about how good 
the normalization performed and whether the images given were good enough.  

It removes some of the photos from dataset, trains the algorithm without them and 
then calculates **accuracy** on these photos. This is repeated a few times and 
the average is shown to user.

##### Manual face recognition
Other examples of using the algorithm are given in [main.py](main.py).  
For example, you can generate images of all the eigenfaces with `save_eigenfaces()` 
or see you own images in the reduced basis with `test_detection_procedure()`.  

Note that all the functions work with already normalized photos.

## Algorithm

##### Normalization
1. **Face detection** was performed using **haar cascade** and _cv_ library. After
detecting, faces where rescaled to `200x200` pixels.
2. Then **gamma correction** and  **histogram equalisation** normalized the lighting
on images.
3. A **gabor filter** was used to further make details more vivid.

##### Training
Training was performed using **PCA**.  
For all the images in dataset, their **weight** (vector in the reduced vector space
of eigenvectors of the covariance matrix) is calucated.

##### Recognition
To recognize the person on a picture, we compared the `weight` of face on image to the 
`average weigh`t of images of every person in dataset.  

The comparison was performed based on the **distance** between the `weight vectors`. The person with
the **lowest distance** was chosen.  

To produce the percentages for each person, we decided to approximate the
distance from person's `weight vector` to the `average vector` using _normal distribution_. 
Then we used the cdf to get the probability. If the distance is much larger than its expected
value, the probability is low. If the distance is much lower, than its almost definitely that
person.

## Results

The algorithm produced accuracy of **81%** on the dataset in the end and was
able to detect the person in real-life conditions using the website.
