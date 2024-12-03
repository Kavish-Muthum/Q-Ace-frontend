
module.exports = { get_company_interview_questions };

export async function getCompanyInterviewQuestions(company_name) {
    const PERPLEXITY_API_KEY = import.meta.env.PERPLEXITY_API_KEY;
    const analysis_url = "https://api.perplexity.ai/chat/completions";
    const analysis_headers = {
        "Authorization": `Bearer ${PERPLEXITY_API_KEY}`,
        "Content-Type": "application/json"
    };
    
    const analysis_data = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant that provides common interview questions for specific companies."
            },
            {
                "role": "user",
                "content": `What are 10 common interview questions asked at ${company_name}? Please provide only the questions, one per line, without numbering.`
            }
        ]
    };

    try {
        const response = await fetch(analysis_url, {
            method: 'POST',
            headers: analysis_headers,
            body: JSON.stringify(analysis_data)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if ('choices' in data && data.choices.length > 0) {
            const questions = data.choices[0].message.content.trim().split('\n');
            return questions.filter(q => q.trim());
        }
        return [];
    } catch (error) {
        console.error('Error fetching interview questions:', error);
        return [];
    }
}