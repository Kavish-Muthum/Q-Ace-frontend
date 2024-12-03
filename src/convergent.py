import requests
from openai import OpenAI

# Placeholder for your Perplexity API key
PERPLEXITY_API_KEY = "key here"

# Initialize OpenAI client
client = OpenAI()

company_name = "Red Bull" 
job_title = "Intern - Data Science"

# Transcribe the audio using Whisper
audio_file = open("yash_sample.mp3", "rb")
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)

analysis_url = "https://api.perplexity.ai/chat/completions"
analysis_headers = {
    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
    "Content-Type": "application/json"
}

analysis_data = {
    "model": "llama-3.1-sonar-small-128k-online",
    "messages": [
        {"role": "system", "content": "You are an AI assistant that provides job descriptions through searching the internet thoroughly and accurately after recieving the job title and company."},
        {"role": "user", "content": f"Job title: " + job_title + " Company Name: " + company_name}
    ]
}

response = requests.post(analysis_url, headers=analysis_headers, json=analysis_data)
job_desc = ""
analysis_data = response.json()

if 'choices' in analysis_data and len(analysis_data['choices']) > 0:
        analysis = analysis_data['choices'][0]['message']['content']
        job_desc = analysis

# Analyze the transcription using Perplexity API
analysis_url = "https://api.perplexity.ai/chat/completions"
analysis_headers = {
    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
    "Content-Type": "application/json"
}

analysis_data = {
    "model": "llama-3.1-sonar-small-128k-online",
    "messages": [
        {"role": "system", "content": f"""
        You are a text analyzer who specializes in scrutinizing job descriptions and seeing if the answers somebody gives to potential interview questions align with the job's values. 
        You will be given a job description, interview question, and someone giving a sample response. 
        Your job is to evaluate how well the response answers the question and aligns with the job description. 
        If they mention a company name, look up interview questions that are typically asked and their responses for that company and give tips to how well the user's response aligns with those actions. 
        Always give constructive criticism to the user.
        Additionally, give a user score from 1 to 100 and tell the user how well they did and explain why they got that score.
        """},
        {"role": "user", "content": f"""
        Job Description: {job_desc}
        Interview Question: What skills do you have that will be relevant to this position?
        User Response: {transcription.text}
        """}
    ]
}

analysis_response = requests.post(analysis_url, headers=analysis_headers, json=analysis_data)

# Check if the request was successful
if analysis_response.status_code == 200:
    analysis_data = analysis_response.json()
    
    # Check if 'choices' key exists in the response
    if 'choices' in analysis_data and len(analysis_data['choices']) > 0:
        analysis = analysis_data['choices'][0]['message']['content']
        print(analysis)
    else:
        print("Error: Unexpected response structure from Perplexity API")
        print("Response content:", analysis_data)
else:
    print(f"Error: API request failed with status code {analysis_response.status_code}")
    print("Response content:", analysis_response.text)

import librosa
import numpy as np

# Load the audio file
audio_path = "yash_sample.mp3"
y, sr = librosa.load(audio_path)

# Extract pitch using librosa's piptrack
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

pitch_values = []
magnitude_threshold = np.percentile(magnitudes, 75)

for t in range(pitches.shape[1]):
    valid_indices = np.where(magnitudes[:, t] > magnitude_threshold)[0]
    if len(valid_indices) > 0:
        best_index = valid_indices[np.argmax(magnitudes[valid_indices, t])]
        pitch = pitches[best_index, t]
        if 50 < pitch < 2000:
            pitch_values.append(pitch)

if pitch_values:
    normalized_pitch_values = (np.array(pitch_values) - np.min(pitch_values)) / (np.max(pitch_values) - np.min(pitch_values))
    pitch_variation = np.var(normalized_pitch_values)
    print(f"Normalized Pitch Variation: {pitch_variation:.4f}")
    print(f"Mean Pitch: {np.mean(pitch_values):.2f} Hz")
    print(f"Min Pitch: {np.min(pitch_values):.2f} Hz")
    print(f"Max Pitch: {np.max(pitch_values):.2f} Hz")
else:
    print("No valid pitch values found.")

# Calculate the Root Mean Square (RMS) for loudness
rms = librosa.feature.rms(y=y)[0]
smoothed_rms = librosa.util.normalize(np.convolve(rms, np.ones(5) / 5, mode='same'))
average_volume = smoothed_rms.mean()
print(f"Average Volume (RMS): {average_volume:.4f}")

# Detect onsets for speaking rate
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
speaking_rate = len(onset_times) / librosa.get_duration(y=y, sr=sr)
print(f"Speaking Rate: {speaking_rate:.2f} syllables/sec")

# Pause detection
silence_threshold = 0.075 * smoothed_rms.max()
frames_per_second = sr / 512
min_pause_frames = int(0.5 * frames_per_second)
pauses = smoothed_rms < silence_threshold

pause_count = 0
pause_duration = 0
current_pause_length = 0

for is_pause in pauses:
    if is_pause:
        current_pause_length += 1
    else:
        if current_pause_length >= min_pause_frames:
            pause_count += 1
            pause_duration += current_pause_length / frames_per_second
        current_pause_length = 0

if current_pause_length >= min_pause_frames:
    pause_count += 1
    pause_duration += current_pause_length / frames_per_second

print(f"Number of Pauses (≥0.5 seconds): {pause_count}")
print(f"Total Pause Duration: {pause_duration:.2f} seconds")

# Calculate additional metrics
total_duration = librosa.get_duration(y=y, sr=sr)
speech_duration = total_duration - pause_duration
speech_ratio = speech_duration / total_duration

print(f"Total Audio Duration: {total_duration:.2f} seconds")
print(f"Speech Duration: {speech_duration:.2f} seconds")
print(f"Speech to Total Ratio: {speech_ratio:.2f}")

# Spectral centroid for voice "brightness"
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
mean_spec_cent = np.mean(spec_cent)
print(f"Mean Spectral Centroid: {mean_spec_cent:.2f} Hz")

# Zero-crossing rate for voice "roughness"
zcr = librosa.feature.zero_crossing_rate(y=y)[0]
mean_zcr = np.mean(zcr)
print(f"Mean Zero-Crossing Rate: {mean_zcr:.4f}")

def summarize_speech_analysis(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration, total_duration):
    summary = []
    
    # Analyze pitch variation
    summary.append(f"Pitch Variation: {pitch_variation:.4f}")
    if pitch_variation > 0.1:
        summary.append("This high pitch variation indicates an expressive and engaging tone, which is excellent for an interview. It shows enthusiasm and can help maintain the interviewer's interest.")
    elif 0.05 <= pitch_variation <= 0.1:
        summary.append("This moderate pitch variation is good for an interview setting. It demonstrates a balanced tone that's neither monotonous nor overly dramatic.")
    else:
        summary.append("The low pitch variation might be perceived as monotonous in an interview. Try to add more vocal variety to engage the interviewer better.")

    # Analyze average volume
    summary.append(f"Average Volume: {average_volume:.4f}")
    if average_volume > 0.08:
        summary.append("The high average volume projects confidence, which is great for an interview. Ensure it's not too loud to avoid seeming aggressive.")
    elif 0.04 <= average_volume <= 0.08:
        summary.append("This moderate volume is ideal for an interview. It suggests clear and comfortable communication without being overbearing.")
    else:
        summary.append("The low volume might be perceived as lack of confidence in an interview. Try to speak up a bit to ensure you're heard clearly.")

    # Analyze speaking rate
    summary.append(f"Speaking Rate: {speaking_rate:.2f} syllables/sec")
    if speaking_rate > 4:
        summary.append("Your speaking rate is quite fast for an interview. While it shows enthusiasm, try to slow down a bit to ensure clarity and give the interviewer time to process your responses.")
    elif 2.5 <= speaking_rate <= 4:
        summary.append("This speaking rate is ideal for an interview. It allows for clear communication and gives the interviewer time to follow your responses.")
    else:
        summary.append("Your speaking rate is on the slower side for an interview. While it may convey thoughtfulness, try to pick up the pace slightly to maintain engagement.")

    # Analyze pauses
    summary.append(f"Number of Pauses: {pause_count}")
    summary.append(f"Total Pause Duration: {pause_duration:.2f} seconds")
    pause_frequency = pause_count / (total_duration / 60)  # pauses per minute
    if pause_frequency > 12 or pause_duration / total_duration > 0.3:
        summary.append("You're using frequent or lengthy pauses. While some pauses are good for emphasis, too many might make you appear hesitant. Try to reduce pause frequency slightly.")
    elif 6 <= pause_frequency <= 12 and pause_duration / total_duration <= 0.3:
        summary.append("Your use of pauses is good for an interview. It adds a natural rhythm to your responses and gives you time to think without excessive silence.")
    else:
        summary.append("You're using relatively few pauses. While this can convey confidence, make sure to include some pauses for emphasis and to give the interviewer time to absorb your responses.")

    # Overall assessment
    summary.append("\nOverall Assessment:")
    strengths = []
    areas_for_improvement = []

    if pitch_variation > 0.05:
        strengths.append("expressive tone")
    else:
        areas_for_improvement.append("increase vocal variety")

    if 0.04 <= average_volume <= 0.08:
        strengths.append("appropriate volume")
    elif average_volume < 0.04:
        areas_for_improvement.append("speak up a bit")
    else:
        areas_for_improvement.append("moderate your volume slightly")

    if 2.5 <= speaking_rate <= 4:
        strengths.append("good speaking pace")
    elif speaking_rate > 4:
        areas_for_improvement.append("slow down slightly")
    else:
        areas_for_improvement.append("increase your speaking pace a bit")

    if 6 <= pause_frequency <= 12 and pause_duration / total_duration <= 0.3:
        strengths.append("effective use of pauses")
    elif pause_frequency > 12 or pause_duration / total_duration > 0.3:
        areas_for_improvement.append("reduce pause frequency")
    else:
        areas_for_improvement.append("include more strategic pauses")

    if strengths:
        summary.append("Strengths: " + ", ".join(strengths) + ".")
    if areas_for_improvement:
        summary.append("Areas for improvement: " + ", ".join(areas_for_improvement) + ".")

    return "\n".join(summary)



# Example usage (you would need to call this function with the actual values)
total_duration = librosa.get_duration(y=y, sr=sr)
print(summarize_speech_analysis(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration, total_duration))


def calculate_audio_score(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration, total_duration, speech_ratio):
    # Pitch Variation Score
    pitch_score = 25 * (1 - min(abs(pitch_variation - 0.075) / 0.075, 1))
    
    # Average Volume Score
    volume_score = 20 * (1 - min(abs(average_volume - 0.06) / 0.06, 1))
    
    # Speaking Rate Score
    rate_score = 20 * (1 - min(abs(speaking_rate - 3.25) / 3.25, 1))
    
    # Pause Usage Score
    pause_frequency = pause_count / (total_duration / 60)
    pause_duration_ratio = pause_duration / total_duration
    pause_score = 10 * (1 - min(abs(pause_frequency - 9) / 9, 1)) + 10 * (1 - min(abs(pause_duration_ratio - 0.2) / 0.2, 1))
    
    # Speech to Total Ratio Score
    ratio_score = 15 * (1 - min(abs(speech_ratio - 0.8) / 0.8, 1))
    
    # Total Score
    total_score = pitch_score + volume_score + rate_score + pause_score + ratio_score
    
    return round(total_score)

# Calculate the score using the extracted metrics
audio_score = calculate_audio_score(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration, total_duration, speech_ratio)

print(f"Overall Audio Score: {audio_score}/100")

# Function to get common interview questions for a company
def get_company_interview_questions(company_name):
    analysis_url = "https://api.perplexity.ai/chat/completions"
    analysis_headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    analysis_data = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "system", "content": "You are an AI assistant that provides common interview questions for specific companies."},
            {"role": "user", "content": f"What are 10 common interview questions asked at {company_name}? Please provide only the questions, one per line, without numbering."}
        ]
    }
    
    response = requests.post(analysis_url, headers=analysis_headers, json=analysis_data)
    
    if response.status_code == 200:
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0:
            questions = data['choices'][0]['message']['content'].strip().split('\n')
            return [q.strip() for q in questions if q.strip()]
    
    return []

# Get common interview questions
common_questions = get_company_interview_questions(company_name)

# Add this to your existing analysis_data

print(f"\nCommon Interview Questions for {company_name}:")
print(common_questions)