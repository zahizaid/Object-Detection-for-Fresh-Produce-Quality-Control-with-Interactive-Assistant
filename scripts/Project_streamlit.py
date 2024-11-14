#%%
# import 
from ultralytics import solutions, YOLO
import streamlit as st
import requests
import requests
from PIL import Image
import cv2  # OpenCV for webcam access
import numpy as np
import io
#%%
model = YOLO(r"C:\Users\zahiz\Desktop\yolo_capstone5\Sayuran 2.0.v12i.yolov11\runs\detect\train\weights\best.pt")
#%%
#CONSTANT
BASE_API_URL = "http://127.0.0.1:7860"
FLOW_ID = "2473a72e-2855-449d-b801-5764fed61811"
ENDPOINT = "" # You can set a specific endpoint name in the flow settings

# #%%
# def run_flow(message : str) -> dict:
# # Run a flow with a given message

# # param message: The message to send to the flow
# # :return: The JSON response from the flow

#     api_url = f"{BASE_API_URL}/api/v1/run/{ENDPOINT or FLOW_ID}"

#     payload = {
#         "input_value": message,
#         "output_type": "chat",
#         "input_type": "chat",
#     }

#     response = requests.post(api_url, json=payload)
#     return response.json()

# def extract_message(response: dict) -> str:

#     try:
#         return response['outputs'][0]['outputs'][0]['results']['message']['text']
#     except (KeyError, IndexError):
#         return "No valid message found in response."
# # Streamlit App
# def main():
#     st.title("AI Chatbot")
#     st.markdown("![Alt Text](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMW83cmExejBocXNqOXF1dzNjNXU3Z3VnYW02ZTgwcDZqejdiaG5qcCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3ornk8GpxBL3AFvdXG/giphy.gif)")
#     #st.image("cat.jpg",width = 50)
#     st.header("Ask anything")

#     # Initialize session state for chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display previous messages with avatars
#     for message in st.session_state.messages:
#         with st.chat_message(message['role'], avatar=message["avatar"]):
#              st.write(message["content"])

#     if query := st.chat_input("Ask me anything..."):
#         # Add user message to session state
#         st.session_state.messages.append(
#             {
#                 "role":"user",
#                 "content": query,
#                 "avatar":"🥷🏽", #emoji for user
#             }
#         )   
#         with st.chat_message("user",avatar="☁️"): # Display user message
#             st.write(query) 

#         # Call the langflow API and get the assistant's response
#         with st.chat_message("assistant", avatar="☁️"): # Emoji for assistant
#             message_placeholder= st.empty() # placeholder for assistant response
#             with st.spinner("Thinking..."):
#             # Fetch response from langflow
#                 assistant_response = extract_message(run_flow(query))
#                 message_placeholder.write(assistant_response)

#         # Add assistant response to session state
#         st.session_state.messages.append(
#             {
#                 "role": "assistant",
#                 "content": assistant_response,
#                 "avatar":"🤹🏾‍♂️", # Emoji for assistant
#             }
#         )

# if __name__== "__main__":
#     main()


#%% Functions for Langflow
def run_flow(message: str) -> dict:
    # Run a flow with a given message
    api_url = f"{BASE_API_URL}/api/v1/run/{ENDPOINT or FLOW_ID}"
    payload = {
        "input_value": message,
        "output_type": "chat",
        "input_type": "chat",
    }
    response = requests.post(api_url, json=payload)
    return response.json()

def extract_message(response: dict) -> str:
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        return "No valid message found in response."

#%% Streamlit App
#%% Functions for Langflow
def run_flow(message: str) -> dict:
    # Run a flow with a given message
    api_url = f"{BASE_API_URL}/api/v1/run/{ENDPOINT or FLOW_ID}"
    payload = {
        "input_value": message,
        "output_type": "chat",
        "input_type": "chat",
    }
    response = requests.post(api_url, json=payload)
    return response.json()

def extract_message(response: dict) -> str:
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        return "No valid message found in response."

#%% Functions for Langflow
def run_flow(message: str) -> dict:
    # Run a flow with a given message
    api_url = f"{BASE_API_URL}/api/v1/run/{ENDPOINT or FLOW_ID}"
    payload = {
        "input_value": message,
        "output_type": "chat",
        "input_type": "chat",
    }
    response = requests.post(api_url, json=payload)
    return response.json()

def extract_message(response: dict) -> str:
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        return "No valid message found in response."

#%% Streamlit App
def main():
    st.title("AI Chatbot with Object Detection")
    st.markdown("![Alt Text](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMW83cmExejBocXNqOXF1dzNjNXU3Z3VnYW02ZTgwcDZqejdiaG5qcCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3ornk8GpxBL3AFvdXG/giphy.gif)")
    st.header("Upload an image or use the webcam for real-time vegetable detection")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages with avatars
    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message["avatar"]):
            st.write(message["content"])

    # Webcam feed option
    use_webcam = st.checkbox("Use Webcam for Real-Time Detection")

    # Webcam Logic
    if use_webcam:
        # Open the webcam
        video_capture = cv2.VideoCapture(0)
        stframe = st.empty()  # Placeholder for video frames

        while video_capture.isOpened():
            # Read a frame from the webcam
            ret, frame = video_capture.read()
            if not ret:
                st.error("Failed to capture video frame")
                break

            # Convert the frame to RGB format for the model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Perform prediction
            results = model(image)

            # Plot the results on the frame
            result_image = results[0].plot() if hasattr(results[0], 'plot') else None

            if result_image is not None:
                # Convert back to BGR for OpenCV to display
                result_image_bgr = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                # Show the frame with detections in Streamlit
                stframe.image(result_image_bgr, channels="BGR", use_container_width=True)
            else:
                st.error("Failed to plot detection results. Check model output compatibility.")
        
        # Release the video capture object
        video_capture.release()
        cv2.destroyAllWindows()

    # Upload Image Logic
    else:
        # Upload and Display Image
        uploaded_image = st.file_uploader("Upload an Image of a Vegetable", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Perform prediction if model loaded successfully
            if model:
                with st.spinner("Detecting objects..."):
                    try:
                        # Run the YOLO model on the uploaded image
                        results = model(image)

                        # Check if results is a valid list or iterable and has elements
                        if results and isinstance(results, list) and len(results) > 0:
                            # Extract the first result and plot it
                            result_plot = results[0].plot() if hasattr(results[0], 'plot') else None

                            if result_plot is not None:
                                # Display the prediction result
                                st.image(result_plot, caption="Detection Result", use_container_width=True)
                            else:
                                st.error("Failed to plot detection results. Check model output compatibility.")
                        else:
                            st.error("No valid detection results returned from the model.")
                    except Exception as e:
                        st.error(f"Error during model prediction: {e}")
            else:
                st.error("Model is not loaded. Please check the model path and ensure it's correct.")

    # Chat Input Section
    if query := st.chat_input("Ask me anything..."):
        # Add user message to session state
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
                "avatar": "🥷🏽",  # emoji for user
            }
        )
        with st.chat_message("user", avatar="☁️"):  # Display user message
            st.write(query) 

        # Call the Langflow API and get the assistant's response
        with st.chat_message("assistant", avatar="☁️"):  # Emoji for assistant
            message_placeholder = st.empty()  # placeholder for assistant response
            with st.spinner("Thinking..."):
                # Fetch response from Langflow
                assistant_response = extract_message(run_flow(query))
                message_placeholder.write(assistant_response)

        # Add assistant response to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "avatar": "🤹🏾‍♂️",  # Emoji for assistant
            }
        )

if __name__ == "__main__":
    main()
# %%
st.sidebar.image(r"C:\Users\zahiz\Desktop\yolo_capstone5\Sayuran 2.0.v12i.yolov11\IMG_1873.jpg", caption="Ahmad Zahi Bin Mohd Zaid [Project Owner]", use_container_width=True)
st.sidebar.title("Project Documentation")
st.sidebar.markdown("### Capstone Project: Custom YOLO + Langflow ChatBot")
st.sidebar.markdown("""
This project is designed for real-time vegetable quality control using AI-based object detection and an interactive assistant.
The project integrates a YOLO-based model for vegetable classification and a chatbot assistant to provide users with project insights.
""")

st.sidebar.subheader("Introduction")
st.sidebar.markdown("""
The project leverages the YOLO object detection model to identify different types of vegetables, along with a chatbot 
assistant powered by Langflow and RAG (Retrieval-Augmented Generation) for answering questions regarding the project.
""")

st.sidebar.subheader("Model Details")
st.sidebar.markdown("""
The YOLO-based object detection system identifies various vegetables in images with a target mean average precision (mAP) of 0.75 and 
an inference time under 100ms per image. YOLOv8 or YOLOv11 architecture is used as the base, with transfer learning applied from COCO weights.
""")

st.sidebar.subheader("Methodology")
st.sidebar.markdown("""
1. **Data Collection**: Collected a diverse set of vegetable images with various lighting and angles, annotated in YOLO format using Roboflow
2. **Model Training**: Used YOLOv11 with transfer learning and data augmentation to improve accuracy.
3. **Model Deployment**: Implemented in Streamlit for real-time image or webcam-based detection.
""")

st.sidebar.subheader("Dataset")
st.sidebar.markdown("""
- Collected 200 images per class, with 5 classes chosen which is carrot, capsicum, potato, cauliflower and cucumber.
- Images were split into 70% training, 15% validation, and 15% testing sets.
""")

st.sidebar.subheader("Conclusion")
st.sidebar.markdown("""
This project showcases a practical implementation of YOLO for agricultural applications. The integration of a chatbot assistant 
enhances the system's usability by allowing users to query details about the model, dataset, and troubleshooting steps.
""")
# %%
# %%
# %%
# %%
