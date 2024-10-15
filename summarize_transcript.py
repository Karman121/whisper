import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def summarize_transcript(transcript_file):
    # Read the transcript with UTF-8 encoding
    with open(transcript_file, 'r', encoding='utf-8') as file:
        transcript = file.read()

    # Set up the model
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Prompt for summarization and analysis
    prompt = f"""
    Please go through the following transcript and structure it into different sections in chronological order in terms of how they were discussed. Don't miss out on any details. 

    {transcript}

    """

    # Generate the response
    response = model.generate_content(prompt)

    return response.text

# Use the function
transcript_file = 'transcription.txt'  # Make sure this file exists
summary = summarize_transcript(transcript_file)

# Print the summary
print(summary)

# Optionally, save the summary to a file
with open('transcript_summary.txt', 'w', encoding='utf-8') as file:
    file.write(summary)
