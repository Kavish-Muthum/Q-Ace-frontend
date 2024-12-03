export async function POST({ request }) {
    try {
      const data = await request.json();
      const { company } = data;
      
      // Call the Python function from convergent.py
      const questions = await get_company_interview_questions(company);
      
      return new Response(JSON.stringify({ questions }), {
        status: 200,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: {
          'Content-Type': 'application/json'
        }
      });
    }
  }