# VideoSummarizer
A video summarizer as part of my MSc in Artificial Intelligence Final Project

### Description
This project takes as an input a video, and uses it audio to produce a transcription using Google's Speech-To-Text API. It then then presents 6 summarization methods to choose and produce the summary, based on the summarization ratio provided.

### Instructions - How to install and use it
1. Download the project to a local directory
2. Install the dependencies by entering the following command into the command prompt:
  `pip install -r requirements.txt`
3. Install one more dependency that requires conda, using the following command: `conda install scipy`
4. Run the application with *Streamlit* to use the GUI with the following command:
  `streamlit run video_summarizer.py`

### Screenshots of GUI
![](screenshots/gui-1.png)
![](screenshots/gui-2.png)
![](screenshots/gui-3.png)
![](screenshots/gui-4.png)
