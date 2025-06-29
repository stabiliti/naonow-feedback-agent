# main.py
# This is the Python code for your Cloud Function.

import functions_framework
from google.cloud import speech
from google.cloud import storage
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- Configuration ---
# You can leave these as they are if you followed the naming convention.
# Make sure to replace [your-initials] with the same initials you used for your bucket names.
REPORTS_BUCKET_NAME = "esl-feedback-reports-cloudgens"
PROJECT_ID = "esl-feedback-agent" # Use your Project ID
LOCATION = "us-central1" # Use the same region as your function

# This function is triggered when a file is uploaded to the video bucket.
@functions_framework.cloud_event
def esl_video_analyzer(cloud_event):
    """
    This function is triggered by a video upload to a GCS bucket.
    It transcribes the video, analyzes it with Gemini, and saves a report.
    """
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]

    print(f"Processing file: {file_name} from bucket: {bucket_name}.")

    # --- Step 1: Transcribe the Video using Speech-to-Text ---
    gcs_uri = f"gs://{bucket_name}/{file_name}"
    transcript = transcribe_video(gcs_uri)

    if not transcript:
        print("Could not transcribe video. Aborting.")
        return

    print("Successfully transcribed video.")

    # --- Step 2: Analyze Transcript with Gemini ---
    feedback_report = generate_feedback_report(transcript)

    if not feedback_report:
        print("Could not generate feedback report. Aborting.")
        return

    print("Successfully generated feedback report.")

    # --- Step 3: Save the Report to the other bucket ---
    save_report(file_name, feedback_report)

    print(f"Process complete for {file_name}.")


def transcribe_video(gcs_uri: str) -> str:
    """Transcribes the audio from a video file stored in GCS."""
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)

    # Configure Speech-to-Text for video transcription
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP4, # Adjust if you use other formats like MOV
        sample_rate_hertz=16000, # A common sample rate
        language_code="en-US",
        enable_automatic_punctuation=True,
        use_enhanced=True,
        model="video",
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    print("Waiting for transcription to complete...")
    response = operation.result(timeout=900) # Timeout after 15 minutes

    # Concatenate all the transcript parts
    transcript_parts = [result.alternatives[0].transcript for result in response.results]
    return "\n".join(transcript_parts)


def generate_feedback_report(transcript: str) -> str:
    """Uses Gemini to generate a feedback report from a transcript."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    model = GenerativeModel("gemini-1.5-flash-001")

    # This is the detailed prompt from our design document.
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


def save_report(original_file_name: str, report_text: str):
    """Saves the generated report to the reports GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(REPORTS_BUCKET_NAME)

    # Create a name for the report file based on the original video name
    report_file_name = f"feedback-for-{original_file_name}.txt"

    blob = bucket.blob(report_file_name)
    blob.upload_from_string(report_text, content_type="text/plain")

    print(f"Report saved to gs://{REPORTS_BUCKET_NAME}/{report_file_name}")
