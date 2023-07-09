# Complex Custom GPT JARVIS Chatbot Interface (Active Development)

<h2> Active Developement Notice: </h2>
Welcome to this active development project by Alchemic Technology. Please note that this is a work in progress, and as such, you may encounter several reference files and potential errors. We encourage you to build upon and reference sections as you see fit.

<h2> Languages Used: </h2>

- Python
- Javascript
- CSS
- HTML

<h2> Project Overview and Goals: </h2> 

The ultimate vision for this project is to create a highly customizable, user-friendly chatbot that allows users to upload their own files for embedding, vectorizing, and storing in a personal database for easy retrieval. This includes seamless integration with Google Drive for direct file uploads and the ability to remember the entirety of your conversation for a more personalized interaction. We aim to enhance the chatbot's capabilities by enabling web browsing, fetching and processing information from the internet, and future plans include image interpretation capabilities. We're also exploring the integration of different language models for user flexibility. Additionally, we aspire to link the chatbot to your own Notion databases, further expanding its functionality. Please note, while significant progress has been made, not all desired functionalities are available in the current codebase, but we're continuously working to enhance the system and bring this vision to life.

<img src= "https://i.imgur.com/8RMadQO.jpg"/>
<br />

<img src= "https://i.imgur.com/xZBfn9S.jpg"/>
<br />

<h2> Project Structure: </h2> 

The project is organized into several key files and folders, each serving a specific purpose:

<h2> Python Files: </h2>

- JARVIS Working Base Model Folder: This folder contains a rough interface for a chatbot with the ability to upload your own documents (not linked to Google Drive, embeddings, or DBs). Run optimizedapp.py

- Several embedding and indexing.py: These files contain the functions and code related to generating embeddings and creating an index (if required) for your dataset. It includes functions for loading data, preprocessing, embedding generation, and indexing.

- response_chatbot.py: This file contains the functions and code for interacting with the chatbot, such as sending user input to the model and receiving responses. It includes functions for processing user input, calling the appropriate APIs (like GPT-3.5 Turbo), and formatting the responses.

- utils.py: This file contains utility functions, such as tokenization, cosine similarity, etc., that can be used by both embedding_and_indexing.py and response_chatbot.py.

- googledrive.py and quickstart.py: These files contain the code for integrating with Google Drive.

- setting up a web crawler.py: This file contains code for the setup of a web crawler.

<h2> HTML Templates: </h2>

The project includes several HTML templates for various functionalities of the chatbot:

- Chatbot interface
- Index page
- File listing
- Picker callback
- File upload

<h2> JSON Files: </h2>

The project uses JSON files for various purposes:

- credentials.json: Contains the credentials for various services.
- embeddings.json: Stores the embeddings generated by the model.
- index.json: Contains the index created for the dataset.

<h2> Docker Folder: </h2>

The Docker folder contains the setup for MindsDB, a machine learning tool.

<h2> Contributing: </h2>

We welcome contributions to this project. Feel free to submit pull requests or raise issues if you encounter any problems or have suggestions for improvements.

<h2> Contact: </h2>

For any queries or issues, please raise an issue on this repository, and we'll do our best to assist you.

Sincerely,

Alchemic Technology
