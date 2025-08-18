import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class RecommendationGenerator:
    def __init__(self):
        """Initialize Groq client."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        try:
            self.client = Groq(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {str(e)}")
    
    def generate_recommendation(self, predicted_class, retrieved_info, confidence):
        """Generate recommendation based on predicted class and retrieved info."""
        try:
            prompt = f"""
            The crop disease '{predicted_class}' was detected with {confidence:.2%} confidence.
            Relevant information: {retrieved_info}
            Provide a concise, actionable recommendation for a farmer to manage this disease.
            """
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating recommendation: {str(e)}"

if __name__ == "__main__":
    try:
        generator = RecommendationGenerator()
        recommendation = generator.generate_recommendation(
            predicted_class="Apple___Cedar_apple_rust",
            retrieved_info="Apply fungicides like myclobutanil or sulfur during early spring. Remove nearby cedar trees to reduce spore spread.",
            confidence=0.8637
        )
        print(recommendation)
    except Exception as e:
        print(f"Error: {str(e)}")