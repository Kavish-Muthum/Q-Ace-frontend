import OpenAI from 'openai';
import axios from 'axios'
import fs from 'fs'
import librosa from 'librosa-js'

module.exports = { get_company_interview_questions };

// Placeholder for your Perplexity API key
const PERPLEXITY_API_KEY = "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";

// Initialize OpenAI client
const client = new OpenAI();

const company_name = "Red Bull";
const job_title = "Intern - Data Science";

// Transcribe the audio using Whisper
async function transcribeAudio() {
  const audio_file = fs.createReadStream("yash_sample.mp3");
  const transcription = await client.audio.transcriptions.create({
    model: "whisper-1",
    file: audio_file
  });
  return transcription.text;
}

async function getJobDescription() {
  const analysis_url = "https://api.perplexity.ai/chat/completions";
  const analysis_headers = {
    "Authorization": `Bearer ${PERPLEXITY_API_KEY}`,
    "Content-Type": "application/json"
  };
  const analysis_data = {
    "model": "llama-3.1-sonar-small-128k-online",
    "messages": [
      {"role": "system", "content": "You are an AI assistant that provides job descriptions through searching the internet thoroughly and accurately after receiving the job title and company."},
      {"role": "user", "content": `Job title: ${job_title} Company Name: ${company_name}`}
    ]
  };

  const response = await axios.post(analysis_url, analysis_data, { headers: analysis_headers });
  if (response.data.choices && response.data.choices.length > 0) {
    return response.data.choices[0].message.content;
  }
  return "";
}

async function analyzeTranscription(job_desc, transcription) {
  const analysis_url = "https://api.perplexity.ai/chat/completions";
  const analysis_headers = {
    "Authorization": `Bearer ${PERPLEXITY_API_KEY}`,
    "Content-Type": "application/json"
  };
  const analysis_data = {
    "model": "llama-3.1-sonar-small-128k-online",
    "messages": [
      {"role": "system", "content": `You are a text analyzer who specializes in scrutinizing job descriptions and seeing if the answers somebody gives to potential interview questions align with the job's values. You will be given a job description, interview question, and someone giving a sample response. Your job is to evaluate how well the response answers the question and aligns with the job description. If they mention a company name, look up interview questions that are typically asked and their responses for that company and give tips to how well the user's response aligns with those actions. Always give constructive criticism to the user. Additionally, give a user score from 1 to 100 and tell the user how well they did and explain why they got that score.`},
      {"role": "user", "content": `Job Description: ${job_desc}, Interview Question: What skills do you have that will be relevant to this position? User Response: ${transcription}`}
    ]
  };

  const response = await axios.post(analysis_url, analysis_data, { headers: analysis_headers });
  if (response.data.choices && response.data.choices.length > 0) {
    return response.data.choices[0].message.content;
  }
  return "Error: Unexpected response structure from Perplexity API";
}

async function analyzeAudio() {
  const { y, sr } = await librosa.load("yash_sample.mp3");

  // Extract pitch using librosa's piptrack
  const { pitches, magnitudes } = librosa.piptrack(y, sr);
  const pitch_values = [];
  const magnitude_threshold = librosa.util.percentile(magnitudes, 75);

  for (let t = 0; t < pitches[0].length; t++) {
    const valid_indices = magnitudes[0].map((m, i) => m > magnitude_threshold ? i : -1).filter(i => i !== -1);
    if (valid_indices.length > 0) {
      const best_index = valid_indices.reduce((a, b) => magnitudes[0][a] > magnitudes[0][b] ? a : b);
      const pitch = pitches[0][best_index];
      if (pitch > 50 && pitch < 2000) {
        pitch_values.push(pitch);
      }
    }
  }

  if (pitch_values.length > 0) {
    const normalized_pitch_values = librosa.util.normalize(pitch_values);
    const pitch_variation = librosa.util.variance(normalized_pitch_values);
    console.log(`Normalized Pitch Variation: ${pitch_variation.toFixed(4)}`);
    console.log(`Mean Pitch: ${librosa.util.mean(pitch_values).toFixed(2)} Hz`);
    console.log(`Min Pitch: ${Math.min(...pitch_values).toFixed(2)} Hz`);
    console.log(`Max Pitch: ${Math.max(...pitch_values).toFixed(2)} Hz`);
  } else {
    console.log("No valid pitch values found.");
  }

  // Calculate the Root Mean Square (RMS) for loudness
  const rms = librosa.feature.rms(y)[0];
  const smoothed_rms = librosa.util.normalize(librosa.util.convolve(rms, new Array(5).fill(1/5)));
  const average_volume = librosa.util.mean(smoothed_rms);
  console.log(`Average Volume (RMS): ${average_volume.toFixed(4)}`);

  // Detect onsets for speaking rate
  const onset_frames = librosa.onset.onset_detect(y, sr);
  const onset_times = librosa.util.frames_to_time(onset_frames, sr);
  const speaking_rate = onset_times.length / librosa.get_duration(y, sr);
  console.log(`Speaking Rate: ${speaking_rate.toFixed(2)} syllables/sec`);

  // Pause detection
  const silence_threshold = 0.075 * Math.max(...smoothed_rms);
  const frames_per_second = sr / 512;
  const min_pause_frames = Math.floor(0.5 * frames_per_second);
  const pauses = smoothed_rms.map(r => r < silence_threshold);
  let pause_count = 0;
  let pause_duration = 0;
  let current_pause_length = 0;

  pauses.forEach((is_pause, i) => {
    if (is_pause) {
      current_pause_length++;
    } else {
      if (current_pause_length >= min_pause_frames) {
        pause_count++;
        pause_duration += current_pause_length / frames_per_second;
      }
      current_pause_length = 0;
    }
  });

  if (current_pause_length >= min_pause_frames) {
    pause_count++;
    pause_duration += current_pause_length / frames_per_second;
  }

  console.log(`Number of Pauses (≥0.5 seconds): ${pause_count}`);
  console.log(`Total Pause Duration: ${pause_duration.toFixed(2)} seconds`);

  // Calculate additional metrics
  const total_duration = librosa.get_duration(y, sr);
  const speech_duration = total_duration - pause_duration;
  const speech_ratio = speech_duration / total_duration;
  console.log(`Total Audio Duration: ${total_duration.toFixed(2)} seconds`);
  console.log(`Speech Duration: ${speech_duration.toFixed(2)} seconds`);
  console.log(`Speech to Total Ratio: ${speech_ratio.toFixed(2)}`);

  // Spectral centroid for voice "brightness"
  const spec_cent = librosa.feature.spectral_centroid(y, sr)[0];
  const mean_spec_cent = librosa.util.mean(spec_cent);
  console.log(`Mean Spectral Centroid: ${mean_spec_cent.toFixed(2)} Hz`);

  // Zero-crossing rate for voice "roughness"
  const zcr = librosa.feature.zero_crossing_rate(y)[0];
  const mean_zcr = librosa.util.mean(zcr);
  console.log(`Mean Zero-Crossing Rate: ${mean_zcr.toFixed(4)}`);

  return {
    pitch_variation,
    average_volume,
    speaking_rate,
    pause_count,
    pause_duration,
    total_duration,
    speech_ratio
  };
}

function summarize_speech_analysis(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration, total_duration) {
  // ... (same as Python version)
}

function calculate_audio_score(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration, total_duration, speech_ratio) {
  // ... (same as Python version)
}

export async function get_company_interview_questions(company_name) {
    const analysis_url = "https://api.perplexity.ai/chat/completions";
    const analysis_headers = {
      "Authorization": `Bearer ${PERPLEXITY_API_KEY}`,
      "Content-Type": "application/json"
    };
    const analysis_data = {
      "model": "llama-3.1-sonar-small-128k-online",
      "messages": [
        {"role": "system", "content": "You are an AI assistant that provides common interview questions for specific companies."},
        {"role": "user", "content": `What are 10 common interview questions asked at ${company_name}? Please provide only the questions, one per line, without numbering.`}
      ]
    };
  
    try {
      const response = await axios.post(analysis_url, analysis_data, { headers: analysis_headers });
      if (response.data.choices && response.data.choices.length > 0) {
        const questions = response.data.choices[0].message.content.trim().split('\n');
        return questions.filter(q => q.trim());
      }
    } catch (error) {
      console.error("Error fetching interview questions:", error);
    }
    return [];
  }
  
  
  

async function main() {
  const transcription = await transcribeAudio();
  const job_desc = await getJobDescription();
  const analysis = await analyzeTranscription(job_desc, transcription);
  console.log(analysis);

  const audioMetrics = await analyzeAudio();
  const summary = summarize_speech_analysis(
    audioMetrics.pitch_variation,
    audioMetrics.average_volume,
    audioMetrics.speaking_rate,
    audioMetrics.pause_count,
    audioMetrics.pause_duration,
    audioMetrics.total_duration
  );
  console.log(summary);

  const audio_score = calculate_audio_score(
    audioMetrics.pitch_variation,
    audioMetrics.average_volume,
    audioMetrics.speaking_rate,
    audioMetrics.pause_count,
    audioMetrics.pause_duration,
    audioMetrics.total_duration,
    audioMetrics.speech_ratio
  );
  console.log(`Audio Score: ${audio_score}`);

  const interview_questions = await get_company_interview_questions(company_name);
  console.log("Common Interview Questions:");
  interview_questions.forEach(q => console.log(q));
}

main().catch(console.error);