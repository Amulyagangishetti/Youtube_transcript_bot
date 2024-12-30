# Youtube_transcript_bot
Generating a youtube video's transcript in desired language and exporting it as pdf/txt file
<br>
Author- Amulya Gangishetti
## Overview
The **Youtube Transcript Bot** allows users to upload a youtube URL, which is then processed and creates a transcript of desired language of the URL. This application uses LangChain frameowrk of LLMs for processing URL and generating transcript and exporting it as txt/pdf file, and Gradio is used as a UI.

## Features

-Upload a Youtube video's URL for generating transcript and exporting it.
-Process the video's URL on the server side
-The transcript is generated in desired langugae and can be exported as txt/pdf 

## Technologies Used
-**UI:**
  - Gradio
-**API:**
  - youtube-trasncript API
  - groq api

## Installation

### Prerequisites
- Python 3.x installed on your machine.
- Required Python packages (listed in `requirements.txt`).
- Ensure you have the necessary libraries for video processing.

This app is used to only for the youtube videos as it uses Youtube transcript API
