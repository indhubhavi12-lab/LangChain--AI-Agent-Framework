from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.7
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms"
)

chain = prompt | llm

topic = "AI Agents"
response = chain.invoke({"topic": topic})

output_text = f"Input: {topic}\n\nOutput:\n{response.content}"

print(output_text)

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("\nSaved to output.txt")
