Here and below is a description of how interact with the code in the repository. Here are tools for recording datasets from video, training classifiers and autoencoders on them, and applying the resulting models in realtime.

Let's start by importing the dataset recording module `record_data` and understand how to work with it. All of the neccessary modules are in the `tools`.

```python
from tools import record_data
```
The video processing in this repository is based on the [mediapipe](https://developers.google.com/mediapipe) framework and uses several solution from it, such as **Face Mesh**, **Hand and Pose Landmarks**, **Holistic**. These solutions allow us to find AU (*Action Units*) points on the human body, and we, in turn, can write these points as a vector representation.

![Example Action Units](https://mediapipe.dev/images/mobile/holistic_sports_and_gestures_example.gif)

That is, we need to extract AU vectors for further training. But, for example, to solve the problem of gesture recognition, you will not consider facial expressions or the whole pose, so we do not need this data.

Let's write a sample function call to extract data for future right hand gesture recognition.
```python
recorder = record_data.record_data

recorder(output_file="data.csv",
         num_frames=200,
         right_hand_landmarks=True,
         class_name="Super")
```
This call accesses the webcam as a source and if there is a subject's right hand in the frame, it creates a vector from AU and writes it to the file `data.csv`. The parameter `num_frames` specifies how many data instances you want to create in the dataset. The parameter `class_name` is used to mark the data instance with a class. If you don't specify this parameter, you have to enter the name of the class anyway in the dialog box.

**All parameters**:
* `output_file: str`
* `num_frames: int`
* `pose_landmarks: bool`
* `face_landmarks: bool`
* `left_hand_landmarks: bool`
* `right_hand_landmarks: bool`
* `pose_cut: bool` we will talk about this parameter later
* `class_name: str`
* `source: str` or `int`

If you don't have a webcam, or if you need to extract features from a pre-recorded video, specify `source` as the parameter (on default `source=0`). For example:
```python
recorder(output_file="data.csv",
         num_frames=200,
         right_hand_landmarks=True,
         class_name="Super",
         source='anomaly_data/CAM2/Andrey/normal/normal_1.mp4')
```
```bash
22%|██▏ | 43/200 [00:09<00:32, 4.76it/s]
```
In the example above, the program could not recognize 200 frames with the right hand in this video, so it recorded only 43.  

**An important reminder:** if you specify the same name for output file, the data is added to it, not overwritten.

Now let's create a dataset on which we will teach the model to recognize the super and peace geasture sign. To do this, let's run the function twice and specify different class names. Then we can represent these two classes in front of the webcam.
```python
recorder(output_file="geasture_dataset.csv",
         num_frames=200,
         right_hand_landmarks=True,
         class_name="Super")
```
```bash
100%|██████████| 200/200 [00:11<00:00, 17.87it/s]
```
```python
recorder(output_file="geasture_dataset.csv",
         num_frames=200,
         right_hand_landmarks=True,
         class_name="Peace")
```
```bash
100%|██████████| 200/200 [00:16<00:00, 11.97it/s]
```
Now we have a finished dataset with 400 data instances, 200 of each class. Let's start training classifier on this data.
```python
from tools import train
```
There are 5 functions in the `train` module that you need to be aware of. We'll go through them all and start with the simplest one, namely `train_classifier`.

Let's first create a classifier model for our dataset, and then learn in detail how to work with it.
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

simple_trainer = train.train_classifier
pipeline = {
    'RFC': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'GBC': make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

simple_trainer(data_file="geasture_dataset.csv",
               pipelines=pipeline,
               output_name="single_geasture",
               test_size=0.2,
               test=True)
```
The function call lists almost all the parameters it has. The `data_file` specifies which data to use for training. `output_name` will use the prefix for the final model name. `test_size` indicates how much data you will need to test the model (the rest will go for training). `test` is bool parameter indicates whether you will need to test the data with the resulting model and will output the results.

Separately, it is worth talking about the parameter `pipelines`. The points is that you can specify the models on which you want to train your classifier using the implementation from **scikit-learn** `make_pipeline`. It is a dictionary, where the key is the nae of the pipeline (this key will also be used to create a file name), and the value is the pipeline you created.

In this case there will be two output models in the `models` folder, names `geasture_GBC.pkl` and `geasture_RFC.pkl`, which use the corresponding algorithms to solve the classification problem.

There is also such a problem: when it comes to video classification, you may need to consider not only a particular frame for purpose, but also neighboring frames (e.g. previous ones). If you need to condiser not only current but also previous frames, you need the `neighbors` parameter.
```python
simple_trainer(data_file="geasture_dataset.csv",
               pipelines=pipeline,
               output_name="geasture_time",
               test_size=0.2,
               test=True,
               neighbors=[-5, -2])
```
In this parameter you feed the negative integer numbers of the previous frames that you want to take into account in the model. In the example above, the model will consider not only the current frame for classificationm but also the two frames before the current one, namely two frames and five frames bedore the current one.