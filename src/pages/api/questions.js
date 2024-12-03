export const prerender = false;

export async function POST({ request }) {
    const PERPLEXITY_API_KEY = import.meta.env.PERPLEXITY_API_KEY;
    
    try {
        const { company } = await request.json();
        
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
                    "content": `What are 10 common interview questions asked at ${company}? Please provide only the questions, one per line, without numbering.`
                }
            ]
        };

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
            return new Response(JSON.stringify({ 
                questions: questions.filter(q => q.trim()) 
            }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' }
            });
        }
        
        return new Response(JSON.stringify({ questions: [] }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
        });
    } catch (error) {
        return new Response(JSON.stringify({ error: error.message }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}