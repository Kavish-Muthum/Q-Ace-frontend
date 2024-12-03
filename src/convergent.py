import requests
from openai import OpenAI
import librosa
import numpy as np

# Initialize API keys
PERPLEXITY_API_KEY = "your_perplexity_api_key"

# Initialize OpenAI client
client = OpenAI()

def get_company_interview_questions(company_name):
    analysis_url = "https://api.perplexity.ai/chat/completions"
    analysis_headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    analysis_data = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that provides common interview questions for specific companies."
            },
            {
                "role": "user",
                "content": f"What are 10 common interview questions asked at {company_name}? Please provide only the questions, one per line, without numbering."
            }
        ]
    }
    
    response = requests.post(analysis_url, headers=analysis_headers, json=analysis_data)
    
    if response.status_code == 200:
        data = response.json()
        if 'choices' in data and len(data['choices']) > 0:
            questions = data['choices'][0]['message']['content'].strip().split('\n')
            return [q.strip() for q in questions if q.strip()]
    return []

def analyze_audio(audio_path):
    # Load the audio file
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
    else:
        pitch_variation = 0
        
    # Calculate RMS for loudness
    rms = librosa.feature.rms(y=y)[0]
    smoothed_rms = librosa.util.normalize(np.convolve(rms, np.ones(5) / 5, mode='same'))
    average_volume = smoothed_rms.mean()
    
    # Detect onsets for speaking rate
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    speaking_rate = len(onset_times) / librosa.get_duration(y=y, sr=sr)
    
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
        
    total_duration = librosa.get_duration(y=y, sr=sr)
    speech_duration = total_duration - pause_duration
    speech_ratio = speech_duration / total_duration
    
    return {
        'pitch_variation': pitch_variation,
        'average_volume': average_volume,
        'speaking_rate': speaking_rate,
        'pause_count': pause_count,
        'pause_duration': pause_duration,
        'total_duration': total_duration,
        'speech_ratio': speech_ratio
    }

def summarize_speech_analysis(metrics):
    summary = []
    
    # Analyze pitch variation
    if metrics['pitch_variation'] > 0.1:
        summary.append("High pitch variation indicates an expressive and engaging tone.")
    elif metrics['pitch_variation'] >= 0.05:
        summary.append("Moderate pitch variation shows good balance in tone.")
    else:
        summary.append("Consider adding more vocal variety to avoid monotony.")
    
    # Analyze volume
    if metrics['average_volume'] > 0.08:
        summary.append("Volume projects confidence but ensure it's not too loud.")
    elif metrics['average_volume'] >= 0.04:
        summary.append("Volume level is ideal for clear communication.")
    else:
        summary.append("Consider speaking up to project more confidence.")
    
    # Analyze speaking rate
    if metrics['speaking_rate'] > 4:
        summary.append("Consider slowing down to ensure clarity.")
    elif metrics['speaking_rate'] >= 2.5:
        summary.append("Speaking rate is ideal for clear communication.")
    else:
        summary.append("Consider picking up the pace slightly.")
    
    return "\n".join(summary)

def calculate_audio_score(metrics):
    # Scoring weights
    weights = {
        'pitch_variation': 25,
        'volume': 20,
        'speaking_rate': 20,
        'pauses': 20,
        'speech_ratio': 15
    }
    
    # Calculate individual scores
    pitch_score = weights['pitch_variation'] * (1 - min(abs(metrics['pitch_variation'] - 0.075) / 0.075, 1))
    volume_score = weights['volume'] * (1 - min(abs(metrics['average_volume'] - 0.06) / 0.06, 1))
    rate_score = weights['speaking_rate'] * (1 - min(abs(metrics['speaking_rate'] - 3.25) / 3.25, 1))
    
    pause_frequency = metrics['pause_count'] / (metrics['total_duration'] / 60)
    pause_duration_ratio = metrics['pause_duration'] / metrics['total_duration']
    pause_score = weights['pauses'] * (1 - min(abs(pause_frequency - 9) / 9, 1))
    
    ratio_score = weights['speech_ratio'] * (1 - min(abs(metrics['speech_ratio'] - 0.8) / 0.8, 1))
    
    total_score = pitch_score + volume_score + rate_score + pause_score + ratio_score
    return round(total_score)

def main():
    # Example usage
    company_name = "Example Company"
    audio_path = "interview_response.mp3"
    
    # Get interview questions
    questions = get_company_interview_questions(company_name)
    print(f"\nCommon Interview Questions for {company_name}:")
    for q in questions:
        print(f"- {q}")
    
    # Analyze audio
    metrics = analyze_audio(audio_path)
    
    # Generate summary and score
    summary = summarize_speech_analysis(metrics)
    score = calculate_audio_score(metrics)
    
    print("\nSpeech Analysis Summary:")
    print(summary)
    print(f"\nOverall Audio Score: {score}/100")

if __name__ == "__main__":
    main()