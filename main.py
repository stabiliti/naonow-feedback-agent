# main.py
# This is the complete, updated Python code for your multi-user Cloud Function.

import functions_framework
from google.cloud import speech
from google.cloud import storage
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- Configuration ---
# This section is now updated with the bucket name you provided.
# You still need to replace [YOUR_PROJECT_ID] with your actual Project ID.
REPORTS_BUCKET_NAME = "esl-feedback-reports-cloudgens" 
PROJECT_ID = "Esl-Feedback-Agent" # IMPORTANT: Replace with your actual Project ID from Google Cloud.
LOCATION = "us-central1" # Use the same region as your function

# This function is triggered when a file is uploaded to the video bucket.
@functions_framework.cloud_event
def esl_video_analyzer(cloud_event):
    """
    This function is triggered by a video upload to a GCS bucket.
    It transcribes the video, analyzes it with Gemini, and saves a report
    to a user-specific folder.
    """
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_path = data["name"]  # This will be like "uploads/user123/video.mp4"
    
    print(f"New file detected: gs://{bucket_name}/{file_path}")

    # --- NEW: Extract User ID and Filename from the path ---
    try:
        # The path is expected to be "folder/userID/filename"
        # For example: "uploads/abc123xyz/my-lesson-1.mp4"
        parts = file_path.split('/')
        if len(parts) < 3 or not parts[0] == "uploads":
             print(f"File path '{file_path}' is not in the expected 'uploads/userID/filename' format. Aborting.")
             return

        user_id = parts[1]
        file_name = parts[2]
        print(f"Processing file: '{file_name}' for user: '{user_id}'.")
    except IndexError:
        print(f"File path '{file_path}' is not in the expected format 'uploads/userID/filename'. Aborting.")
        return

    # --- Step 1: Transcribe the Video ---
    gcs_uri = f"gs://{bucket_name}/{file_path}"
    transcript = transcribe_video(gcs_uri)
    
    if not transcript:
        print(f"Could not transcribe video for user {user_id}. Aborting.")
        return
    print(f"Successfully transcribed video for user {user_id}.")

    # --- Step 2: Analyze Transcript with Gemini ---
    feedback_report = generate_feedback_report(transcript)

    if not feedback_report:
        print(f"Could not generate report for user {user_id}. Aborting.")
        return
    print(f"Successfully generated report for user {user_id}.")
        
    # --- Step 3: Save the Report to the user's private folder ---
    save_report(user_id, file_name, feedback_report)

    print(f"Process complete for '{file_name}' for user '{user_id}'.")


def transcribe_video(gcs_uri: str) -> str:
    """Transcribes the audio from a video file stored in GCS."""
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP4,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
        use_enhanced=True,
        model="video",
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    print("Waiting for transcription to complete...")
    response = operation.result(timeout=1800) # Timeout after 30 minutes for longer videos

    transcript_parts = [result.alternatives[0].transcript for result in response.results]
    return "\n".join(transcript_parts)


def generate_feedback_report(transcript: str) -> str:
    """Uses Gemini to generate a feedback report from a transcript."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-1.5-flash-001")
    
    prompt = f"""
    You are an expert ESL teaching coach. Your task is to analyze the following transcript from an ESL class and provide a feedback report. The report should be divided into two sections: "Strengths" and "Improvements".

    When analyzing the transcript, please consider the following aspects of effective ESL teaching:

    **For Strengths, look for:**
    * Clear Instructions
    * High Student Talk Time (STT)
    * Positive Reinforcement and Error Correction
    * Engaging Activities
    * Scaffolding and support
    * Concept Checking Questions (CCQs)

    **For Improvements, look for:**
    * Lack of Clarity in instructions
    * Dominating Teacher Talk Time (TTT)
    * Missed Opportunities for Correction
    * Pacing issues (too fast or too slow)
    * Lack of Student Engagement

    Here is the transcript of the ESL class:
    ---
    {transcript}
    ---

    Please generate the feedback report now in Markdown format.
    """

    response = model.generate_content(prompt)
    return response.text


def save_report(user_id: str, original_file_name: str, report_text: str):
    """Saves the generated report to a user-specific folder in the reports bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(REPORTS_BUCKET_NAME)
    
    # Create a user-specific path for the report
    report_file_name = f"reports/{user_id}/feedback-for-{original_file_name}.txt"
    
    blob = bucket.blob(report_file_name)
    blob.upload_from_string(report_text, content_type="text/plain; charset=utf-8")
    
    print(f"Report saved to gs://{REPORTS_BUCKET_NAME}/{report_file_name}")
