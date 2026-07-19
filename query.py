from groq import Groq
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()  # run a function which reads .env and load its content(api key)into memory so python can access it
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")



# --- everything below is now INSIDE ask_question, so it runs fresh for each question passed in ---
def ask_question(question):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")  # connect to the SAME persistent db that ingest.py created
    collection = chroma_client.get_or_create_collection(name="my_document") ## fetch fresh each call, in case ingest_pdf recreated it since last time

    question_embedding = model.encode(question)  # converts qn to same 384 no embedding like chunks

    results = collection.query(  # asking chroma to search which embedding are closest in meaning to qn's embedding
        query_embeddings=[question_embedding],  # our qn is embedded and we are putting it query_embeddings process of chroma
        n_results=5  # return 5 closest matches
    )

    # results["documents"] = [[chunk_A, chunk_B, chunk_C]] -> outer bracket = questions, inner = matches
    # [0] unwraps the outer bracket (picks question 1's results) -> now just [chunk_A, chunk_B, chunk_C]
    # matched_chunks holds ALL 3 chunks together, not just 1
    matched_chunks = results["documents"][0]  # coz we saved chunks in documents in chroma and collection.query will return documents,distances,ids by default(not embeddings unless we specifically ask),so we specifically ask for documents ie chunks
    if not matched_chunks:
        return "I don't have any relevant information stored to answer that."

    context = "\n\n".join(matched_chunks)  # so context is a string where we are putting matched_chunks with a line space between the chunks

    prompt = f"""Answer the question using only the context. If the answer isn't in the context, say you don't know.

Context:{context}

Question:{question}

"""

    response = client.chat.completions.create(  # the actual API call: "here's a conversation, give me a response"
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}  # mark this message coming from user (me) and content is actual text coming from my side
        ]
    )

    return response.choices[0].message.content  # return instead of print, so we can use the answer for many questions


if __name__ == "__main__":  # only runs this test block when query.py is run directly, not when another file imports ask_question from it
    print("=== STEP 9 & 10: Testing multiple questions ===")

    test_questions = [
        "What are Claude 3's capabilities?",
        "What is Anthropic's approach to sustainability?",
        "Does Claude 3 support multiple languages?",
        "What was the color of the sky in 1800?",   # deliberately unrelated - tests if it says "I don't know"
        "What is the capital of France?",             # deliberately unrelated - tests if it says "I don't know"
        "Compare Claude 3 Haiku and Opus in terms of speed and intelligence",   # tests multi-chunk reasoning
        "What year was Claude 3 announced?",                                    # tests a specific fact
        "Is Claude 3 available on AWS?",                                        # tests a specific fact
        "What are the main risks discussed in the safety section?",            # tests a different section of the doc
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {ask_question(q)}")
