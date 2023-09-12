# AuditoryEmotionRecognition

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Example Outputs](#example-outputs)

## Introduction 

Building upon the foundation laid by the TIMNET model, this repository introduces an extended approach tailored for multi-speaker conversations. The original TIMNET was designed to identify emotions in short audio clips from a single speaker. It is refined  to analyze the ebb and flow of emotions in longer conversations involving multiple participants.

For the foundational work on the TIMNET model, please refer to https://arxiv.org/pdf/2211.08233v2.pdf and https://github.com/Jiaxin-Ye/TIM-Net_SER/tree/main. 

## Installation

### Clone the Project
Acquire the project via git:

        $ git clone https://github.com/NurEser/AuditoryEmotionRecognition.git
        $ cd AuditoryEmotionRecognition
        
Or download as an archive and extract it:

        $ wget https://github.com/NurEser/AuditoryEmotionRecognition/archive/refs/heads/main.zip
        $ unzip main.zip
        $ cd AuditoryEmotionRecognition
        
### Setting Up The Environment
Acquire [miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) or [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) and setup a new environment to install the dependencies.

        $ conda create --name myenv
        $ conda activate myenv
        $ conda env update --file environment.yml
        
## Usage

### Requirements

**1. Audio File(audio_file):**  An audio file in the .wav format.  

**2. Duration Text File(duration_file):** A corresponding duration text file with speaker and time interval information.  

**2.Duration File Format:** Each line of the file should contain the start and end time of a speech segment followed by a comma and then the speaker's identifier. The time format should be mm:ss. An exemplary 'duration.txt' file can also be found in the repository. 

**Example:**  
```
00:00 - 00:15 , speaker1  
00:15 - 00:30 , speaker2  
00:30 - 00:45 , speaker1  
00:45 - 01:00 , speaker3  
```

#### Optional(but Recommended)

**3.Choice of the Model(model_type):** Our auditory emotion recognition system offers two distinct models to cater to different use-cases. If not specified, IEMOCAP model is used. 

&nbsp;&nbsp;&nbsp;&nbsp; **Model1:** This model is trained on IEMOCAP dataset and is the default model. It encompasses four primary emotions:  "angry", "happy", "neutral", and "sad" with more precision. 

&nbsp;&nbsp;&nbsp;&nbsp; **Model2:** This model is trained on RAVDE dataset. It identifies a broader range of emotions: "angry", "calm", "disgust", "fear", "happy", "neutral", "sad", and "surprise". To choose RAVDE model, set model_type to 2. 

**4.Version of the Model(fold):** Each model has slightly different versions numbered from 1 to 10. You can specify the version or use the default one. See instructions for example usage.

**5.Output Saving(output):** If specified, output is saved to the desired location, otherwise the plot is shown on the screen in a separate window. See command line instructions for example usage.

### Using the .py Script
For those who prefer the command line, a .py script is provided. This script uses argparse for easy command-line interactions.

#### Command Line Instructions:
```
$ python emotion_recognition.py --audio_file "path/to/your/audiofile.wav" --duration_file "path/to/your/durationfile.txt" --model_type 1 --fold=9  --output "outputfilename"
```

#### For example:

 ```
$ python auditory_emotion_recognition.py --audio_file 'sample.wav' --duration_file 'durations.txt'
```
### Using the Jupyter Notebook

To utilize the model within a Jupyter Notebook, you'll need to call the main function, providing the paths to the audio and duration files as argument. 
```
$ main(record="path_to_audio.wav", file="path_to_duration.txt" model_type= 1, fold=9)
```


## Example Outputs  

 Outputs are saved as 'output.png' 

### &nbsp; &nbsp; &nbsp; Three Speakers
![three_speaker](https://github.com/NurEser/AuditoryEmotionRecognition/assets/30387028/ed1f3e88-dfd5-4523-ad00-7026a73ddf72)


### &nbsp; &nbsp; &nbsp; Two Speakers
![two_speaker](https://github.com/NurEser/AuditoryEmotionRecognition/assets/30387028/51343714-9e81-48a3-8012-f8d7a007504c)

###  &nbsp; &nbsp; &nbsp; Single Speaker
![single_speaker](https://github.com/NurEser/AuditoryEmotionRecognition/assets/30387028/df222ee3-7966-4e43-b62e-c930c14b514d)

#### Extra plots for single speaker
 &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp;![output_single](https://github.com/NurEser/AuditoryEmotionRecognition/assets/30387028/c18e4be2-c9de-459f-8a62-96869065b110)
