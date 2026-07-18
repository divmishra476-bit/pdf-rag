import fitz
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]  # chunk is string which takes 1000 characters from text and it is temporary as the chunk string get stored in chunks list,and then the next string also get stored in chunk list and continues until the loop ends
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # use the path passed in, not a hardcoded filename
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    chunks = chunk_text(full_text)
    all_embeddings = model.encode(chunks)  # all_embedding is a list so len(all_embedding)=len(chunks)

    chroma_client = chromadb.PersistentClient(path="./chroma_db")  # creating a connection to chroma database (persistent -> saved to disk, survives after script ends)

    try:
        chroma_client.delete_collection(name="my_document")  # wipe old data first, so re-running ingest_pdf on a new PDF doesn't clash with old IDs
    except Exception:
        pass  # if the collection doesn't exist yet (first ever run), there's nothing to delete - just continue

    collection = chroma_client.create_collection(name="my_document")  # start fresh every time ingest_pdf runs

    ids = []
    for i in range(len(chunks)):
        ids.append(str(i))  # coz chroma needs the id to be a string ie str(i)

    collection.add(
        documents=chunks,
        embeddings=all_embeddings,
        ids=ids
    )

    print("=== STEP 4: Extraction ===")
    print(f"Total characters extracted: {len(full_text)}")

    print("=== STEP 5: Chunking ===")
    print(f"Number of chunks: {len(chunks)}")

    print("=== STEP 6: Embeddings ===")
    print("Number of embeddings:", len(all_embeddings))  # should be same as len(chunks)

    print("=== STEP 7: Chroma Storage ===")
    print("Number of items stored:", collection.count())

    return len(chunks)  # so the app can show "X chunks processed"


# __name__ is a hidden variable Python auto-sets for every file.
# If I run THIS file directly (python ingest.py), __name__ becomes "__main__" -> condition is True -> code below runs.
# If ANOTHER file just imports ingest_pdf from here (from ingest import ingest_pdf), __name__ becomes "ingest" instead -> condition is False -> code below is skipped.
# So this block only runs when I test ingest.py myself, never when app.py borrows the function from it.
if __name__ == "__main__":  # only runs this when ingest.py is run directly, not when app.py imports ingest_pdf from it
    num_chunks = ingest_pdf("test_doc.pdf") # and storing the no of chunks in it
    print(f"Ingestion complete. {num_chunks} chunks stored. You can now run query.py")