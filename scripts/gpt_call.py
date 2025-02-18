import os
from openai import OpenAI
from dotenv import load_dotenv
from rag_loop import get_context

load_dotenv()

# Load your OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.

client = OpenAI(api_key=OPENAI_API_KEY)

# def _context =
def create_chatbot_response(user_input: str, context: str = ""):
        # context = await retrieved_context(user_input)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": f"Answer the query {user_input} {context}"},
          ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )

        return response.choices[0].message.content

async def retrieved_context(user_input):
   context = await get_context("bert", user_input, "covestro_hr", "content")
   response = create_chatbot_response(user_input, context)
   return response
