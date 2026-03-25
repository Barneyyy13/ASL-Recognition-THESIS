# ASL Recognition System

This repository contains the core LSTM architecture, extracted feature data, and trained PyTorch weights for our real-time Sign Language Recognition thesis project. 

## How to Set Up Your Local Environment

To run the live inference webcam test on your own computer, follow these steps exactly. **Do not** attempt to run the scripts using your global Python installation.

### 1. Clone the Repository
Open your terminal (or Git Bash) and run:
`https://github.com/Barneyyy13/ASL-Recognition-THESIS.git`
`cd ASL-Recognition-THESIS`

### 2. Create the Virtual Environment
Run this command to create an empty environment folder named `venv`:
`python -m venv venv`

### 3. Activate the Environment
You must activate the environment before installing anything. 
* **If you are on Windows (Git Bash / Command Prompt):**
  `source venv/Scripts/activate`
* **If you are on Mac/Linux:**
  `source venv/bin/activate`

*(You will know it worked if you see `(venv)` appear at the beginning of your terminal prompt line).*

### 4. Install the Blueprint Dependencies
Let the `requirements.txt` file build the exact architecture we need:
`pip install -r requirements.txt`

### 5. Run the Live Inference
Once the installation finishes, you can immediately test the trained AI brain using your webcam:
`inference.py`

Press `q` on your keyboard while the webcam window is selected to close the program safely.