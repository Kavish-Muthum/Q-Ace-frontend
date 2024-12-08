import axios from 'axios';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const librosa = require('librosa');
const fs = require('fs');

// Placeholder for your Perplexity API key
const PERPLEXITY_API_KEY = "insert here";

// Initialize OpenAI client
const client = new OpenAI();

const company_name = "Red Bull"; 
const job_title = "Intern - Data Science";

// Transcribe the audio using Whisper
const audio_file = fs.readFileSync("yash_sample.mp3");
const transcription = await client.audio.transcriptions.create({
    model: "whisper-1",
    file: audio_file
});

const analysis_url = "https://api.perplexity.ai/chat/completions";
const analysis_headers = {
    "Authorization": `Bearer ${PERPLEXITY_API_KEY}`,
    "Content-Type": "application/json"
};

let analysis_data = {
    model: "llama-3.1-sonar-small-128k-online",
    messages: [
        { role: "system", content: "You are an AI assistant that provides job descriptions through searching the internet thoroughly and accurately after receiving the job title and company." },
        { role: "user", content: `Job title: ${job_title} Company Name: ${company_name}` }
    ]
};

let response = await axios.post(analysis_url, analysis_data, { headers: analysis_headers });
let job_desc = "";
let analysis_data = response.data;

if (analysis_data.choices && analysis_data.choices.length > 0) {
    let analysis = analysis_data.choices[0].message.content;
    job_desc = analysis;
}

// Analyze the transcription using Perplexity API
analysis_data = {
    model: "llama-3.1-sonar-small-128k-online",
    messages: [
        { role: "system", content: `
        You are a text analyzer who specializes in scrutinizing job descriptions and seeing if the answers somebody gives to potential interview questions align with the job's values. 
        You will be given a job description, interview question, and someone giving a sample response. 
        Your job is to evaluate how well the response answers the question and aligns with the job description. 
        If they mention a company name, look up interview questions that are typically asked and their responses for that company and give tips to how well the user's response aligns with those actions. 
        Always give constructive criticism to the user.
        Additionally, give a user score from 1 to 100 and tell the user how well they did and explain why they got that score.
        ` },
        { role: "user", content: `
        Job Description: ${job_desc}
        Interview Question: What skills do you have that will be relevant to this position?
        User Response: ${transcription.text}
        ` }
    ]
};

let analysis_response = await axios.post(analysis_url, analysis_data, { headers: analysis_headers });

// Check if the request was successful
if (analysis_response.status === 200) {
    analysis_data = analysis_response.data;

    // Check if 'choices' key exists in the response
    if (analysis_data.choices && analysis_data.choices.length > 0) {
        let analysis = analysis_data.choices[0].message.content;
        console.log(analysis);
    } else {
        console.log("Error: Unexpected response structure from Perplexity API");
        console.log("Response content:", analysis_data);
    }
} else {
    console.log(`Error: API request failed with status code ${analysis_response.status}`);
    console.log("Response content:", analysis_response.data);
}

// Load the audio file
const audio_path = "yash_sample.mp3";
const { y, sr } = await librosa.load(audio_path);

// Extract pitch using librosa's piptrack
const { pitches, magnitudes } = await librosa.piptrack({ y, sr });

let pitch_values = [];
const magnitude_threshold = np.percentile(magnitudes, 75);

for (let t = 0; t < pitches.shape[1]; t++) {
    const valid_indices = np.where(magnitudes[:, t] > magnitude_threshold)[0];
    if (valid_indices.length > 0) {
        const best_index = valid_indices[np.argmax(magnitudes[valid_indices, t])];
        const pitch = pitches[best_index, t];
        if (50 < pitch && pitch < 2000) {
            pitch_values.push(pitch);
        }
    }
}

if (pitch_values.length > 0) {
    const normalized_pitch_values = (np.array(pitch_values) - np.min(pitch_values)) / (np.max(pitch_values) - np.min(pitch_values));
    const pitch_variation = np.var(normalized_pitch_values);
    console.log(`Normalized Pitch Variation: ${pitch_variation.toFixed(4)}`);
    console.log(`Mean Pitch: ${np.mean(pitch_values).toFixed(2)} Hz`);
    console.log(`Min Pitch: ${np.min(pitch_values).toFixed(2)} Hz`);
    console.log(`Max Pitch: ${np.max(pitch_values).toFixed(2)} Hz`);
} else {
    console.log("No valid pitch values found.");
}

// Calculate the Root Mean Square (RMS) for loudness
const rms = await librosa.feature.rms({ y })[0];
const smoothed_rms = await librosa.util.normalize(np.convolve(rms, np.ones(5) / 5, 'same'));
const average_volume = smoothed_rms.mean();
console.log(`Average Volume (RMS): ${average_volume.toFixed(4)}`);

// Detect onsets for speaking rate
const onset_frames = await librosa.onset.onset_detect({ y, sr });
const onset_times = await librosa.frames_to_time(onset_frames, { sr });
const speaking_rate = onset_times.length / librosa.get_duration({ y, sr });
console.log(`Speaking Rate: ${speaking_rate.toFixed(2)} syllables/sec`);

// Pause detection
const silence_threshold = 0.075 * smoothed_rms.max();
const frames_per_second = sr / 512;
const min_pause_frames = Math.floor(0.5 * frames_per_second);
const pauses = smoothed_rms < silence_threshold;

let pause_count = 0;
let pause_duration = 0;
let current_pause_length = 0;

for (const is_pause of pauses) {
    if (is_pause) {
        current_pause_length += 1;
    } else {
        if (current_pause_length >= min_pause_frames) {
            pause_count += 1;
            pause_duration += current_pause_length / frames_per_second;
        }
        current_pause_length = 0;
    }
}

if (current_pause_length >= min_pause_frames) {
    pause_count += 1;
    pause_duration += current_pause_length / frames_per_second;
}

console.log(`Number of Pauses (≥0.5 seconds): ${pause_count}`);
console.log(`Total Pause Duration: ${pause_duration.toFixed(2)} seconds`);

// Calculate additional metrics
const total_duration = await librosa.get_duration({ y, sr });
const speech_duration = total_duration - pause_duration;
const speech_ratio = speech_duration / total_duration;

console.log(`Total Audio Duration: ${total_duration.toFixed(2)} seconds`);
console.log(`Speech Duration: ${speech_duration.toFixed(2)} seconds`);
console.log(`Speech to Total Ratio: ${speech_ratio.toFixed(2)}`);

// Spectral centroid for voice "brightness"
const spec_cent = await librosa.feature.spectral_centroid({ y, sr })[0];
const mean_spec_cent = np.mean(spec_cent);
console.log(`Mean Spectral Centroid: ${mean_spec_cent.toFixed(2)} Hz`);

// Zero-crossing rate for voice "roughness"
const zcr = await librosa.feature.zero_crossing_rate({ y })[0];
const mean_zcr = np.mean(zcr);
console.log(`Mean Zero-Crossing Rate: ${mean_zcr.toFixed(4)}`);

function summarize_speech_analysis(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration, total_duration) {
    const summary = [];

    // Analyze pitch variation
    summary.push(`Pitch Variation: ${pitch_variation.toFixed(4)}`);
    if (pitch_variation > 0.1) {
        summary.push("This high pitch variation indicates an expressive and engaging tone, which is excellent for an interview. It shows enthusiasm and can help maintain the interviewer's interest.");
    } else if (0.05 <= pitch_variation && pitch_variation <= 0.1) {
        summary.push("This moderate pitch variation is good for an interview setting. It demonstrates a balanced tone that's neither monotonous nor overly dramatic.");
    } else {
        summary.push("The low pitch variation might be perceived as monotonous in an interview. Try to add more vocal variety to engage the interviewer better.");
    }

    // Analyze average volume
    summary.push(`Average Volume: ${average_volume.toFixed(4)}`);
    if (average_volume > 0.08) {
        summary.push("The high average volume projects confidence, which is great for an interview. Ensure it's not too loud to avoid seeming aggressive.");
    } else if (0.04 <= average_volume && average_volume <= 0.08) {
        summary.push("This moderate volume is ideal for an interview. It suggests clear and comfortable communication without being overbearing.");
    } else {
        summary.push("The low volume might be perceived as lack of confidence in an interview. Try to speak up a bit to ensure you're heard clearly.");
    }

    // Analyze speaking rate
    summary.push(`Speaking Rate: ${speaking_rate.toFixed(2)} syllables/sec`);
    if (speaking_rate > 4) {
        summary.push("Your speaking rate is quite fast for an interview. While it shows enthusiasm, try to slow down a bit to ensure clarity and give the interviewer time to process your responses.");
    } else if (2.5 <= speaking_rate && speaking_rate <= 4) {
        summary.push("This speaking rate is ideal for an interview. It allows for clear communication and gives the interviewer time to follow your responses.");
    } else {
        summary.push("Your speaking rate is on the slower side for an interview. While it may convey thoughtfulness, try to pick up the pace slightly to maintain engagement.");
    }

    // Analyze pauses
    summary.push(`Number of Pauses: ${pause_count}`);
    summary.push(`Total Pause Duration: ${pause_duration.toFixed(2)} seconds`);
    const pause_frequency = pause_count / (total_duration / 60);  // pauses per minute
    if (pause_frequency > 12 || pause_duration / total_duration > 0.3) {
        summary.push("You're using frequent or lengthy pauses. While some pauses are good for emphasis, too many might make you appear hesitant. Try to reduce pause frequency slightly.");
    } else if (6 <= pause_frequency && pause_frequency <= 12 && pause_duration / total_duration <= 0.3) {
        summary.push("Your use of pauses is good for an interview. It adds a natural rhythm to your responses and gives you time to think without excessive silence.");
    } else {
        summary.push("You're using relatively few pauses. While this can convey confidence, make sure to include some pauses for emphasis and to give the interviewer time to absorb your responses.");
    }

    // Overall assessment
    summary.push("\nOverall Assessment:");
    const strengths = [];
    const areas_for_improvement = [];

    if (pitch_variation > 0.05) {
        strengths.push("expressive tone");
    } else {
        areas_for_improvement.push("increase vocal variety");
    }

    if (0.04 <= average_volume && average_volume <= 0.08) {
        strengths.push("appropriate volume");
    } else if (average_volume < 0.04) {
        areas_for_improvement.push("speak up a bit");
    } else {
        areas_for_improvement.push("moderate your volume slightly");
    }

    if (2.5 <= speaking_rate && speaking_rate <= 4) {
        strengths.push("good speaking pace");
    } else if (speaking_rate > 4) {
        areas_for_improvement.push("slow down slightly");
    } else {
        areas_for_improvement.push("increase your speaking pace a bit");
    }

    if (6 <= pause_frequency && pause_frequency <= 12 && pause_duration / total_duration <= 0.3) {
        strengths.push("effective use of pauses");
    } else if (pause_frequency > 12 || pause_duration / total_duration > 0.3) {
        areas_for_improvement.push("reduce pause frequency");
    } else {
        areas_for_improvement.push("include more strategic pauses");
    }

    if (strengths.length > 0) {
        summary.push("Strengths: " + strengths.join(", ") + ".");
    }
    if (areas_for_improvement.length > 0) {
        summary.push("Areas for improvement: " + areas_for_improvement.join(", ") + ".");
    }

    return summary.join("\n");
}

// Example usage (you would need to call this function with the actual values)
const total_duration = await librosa.get_duration({ y, sr });
console.log(summarize_speech_analysis(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration, total_duration));

function calculate_audio_score(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration, total_duration, speech_ratio) {
    // Pitch Variation Score
    const pitch_score = 25 * (1 - Math.min(Math.abs(pitch_variation - 0.075) / 0.075, 1));
    
    // Average Volume Score
    const volume_score = 20 * (1 - Math.min(Math.abs(average_volume - 0.06) / 0.06, 1));
    
    // Speaking Rate Score
    const rate_score = 20 * (1 - Math.min(Math.abs(speaking_rate - 3.25) / 3.25, 1));
    
    // Pause Usage Score
    const pause_frequency = pause_count / (total_duration / 60);
    const pause_duration_ratio = pause_duration / total_duration;
    const pause_score = 10 * (1 - Math.min(Math.abs(pause_frequency - 9) / 9, 1)) + 10 * (1 - Math.min(Math.abs(pause_duration_ratio - 0.2) / 0.2, 1));
    
    // Speech to Total Ratio Score
    const ratio_score = 15 * (1 - Math.min(Math.abs(speech_ratio - 0.8) / 0.8, 1));
    
    // Total Score
    const total_score = pitch_score + volume_score + rate_score + pause_score + ratio_score;
    
    return Math.round(total_score);
}

// Calculate the score using the extracted metrics
const audio_score = calculate_audio_score(pitch_variation, average_volume, speaking_rate, pause_count, pause_duration, total_duration, speech_ratio);

console.log(`Overall Audio Score: ${audio_score}/100`);

// Function to get common interview questions for a company
async function get_company_interview_questions(company_name) {
    const analysis_url = "https://api.perplexity.ai/chat/completions";
    const analysis_headers = {
        "Authorization": `Bearer ${PERPLEXITY_API_KEY}`,
        "Content-Type": "application/json"
    };
    
    const analysis_data = {
        model: "llama-3.1-sonar-small-128k-online",
        messages: [
            { role: "system", content: "You are an AI assistant that provides common interview questions for specific companies." },
            { role: "user", content: `What are 10 common interview questions asked at ${company_name}? Please provide only the questions, one per line, without numbering.` }
        ]
    };
    
    const response = await axios.post(analysis_url, analysis_data, { headers: analysis_headers });
    
    if (response.status === 200) {
        const data = response.data;
        if (data.choices && data.choices.length > 0) {
            const questions = data.choices[0].message.content.trim().split('\n');
            return questions.map(q => q.trim()).filter(q => q);
        }
    }
    
    return [];
}

// Get common interview questions
const common_questions = await get_company_interview_questions(company_name);

console.log(`\nCommon Interview Questions for ${company_name}:`);
console.log(common_questions);