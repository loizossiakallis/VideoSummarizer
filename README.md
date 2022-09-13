# VideoSummarizer
A video summarizer implemented during my MSc in Artificial Intelligence Final Project

## Table of contents:
1. [Description](#des)
2. [Instructions](#instructions)
3. [Video demonstrating the system's functionality](#video)
4. [Screenshots of GUI](#screenshots)

<a name="des"></a>
### Description <a name="des"></a>
This project takes as an input a video, and uses it audio to produce a transcription using Google's Speech-To-Text API. It then then presents 6 summarization methods to choose and produce the summary, based on the summarization ratio provided.

<a name="instructions"></a>
### Instructions - How to install and use it
*Requires Python 3.8 (specifically I used Python 3.10)*
1. Download the project to a local directory
2. Install the dependencies by entering the following command into the command prompt: `pip install -r requirements.txt`
3. Run the application with *Streamlit* to use the GUI with the following command: `streamlit run video_summarizer.py`

<a name="video"></a>
### Video demonstrating the system's functionality (click on the picture below to redirect to the youtube video)
*Note: the transcription process takes the longest so skip to 1:52 where it finishes*
[![Video demonstration](https://img.youtube.com/vi/3FHJODYE1Ps/maxresdefault.jpg)](https://youtu.be/3FHJODYE1Ps)

<a name="screenshots"></a>
### Screenshots of GUI
![](screenshots/gui-1.png)
![](screenshots/gui-2.png)
![](screenshots/gui-3.png)
![](screenshots/gui-4.png)
