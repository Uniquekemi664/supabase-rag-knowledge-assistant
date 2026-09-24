from sentence_transformers import SentenceTransformer
import numpy as np 

# Load the embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Generate Embeddings

def get_embedding(text):
    return model.encode(text)

#Test texts- related and unrelated 
texts = ["parental leave policy for new parents", "maternity and paternity benefits", "How to make pizza at home", "Employee vacation policy", "Pizza delivery near me"]
print()
print("Generating Embedding")
print("="*40)

embeddings = {}
for text in texts:
    embeddings[text] = get_embedding(text)
    print("Embedding:", text)
    print("Vector dimension:", len(embeddings[text]))
    print("First 3 values:", [round(v, 4) for v in embeddings[text][:3]])
    print()

# Compute Similarity

def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product/ (norm1 *norm2)
#Compare parental leave with everything
query_text = "parental leave policy for new parents"
query_embeddings = embeddings[query_text]
print()
print("Similarity to:", query_text)
print("="*40)
print()

for text, embedding in embeddings.items():
    similarity = cosine_similarity(query_embeddings, embedding)
    print(" "+str(round(similarity, 4)) +"-"+ text)

# Build a simple vector Database 
class SimpleVectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = []


    def add(self, text, metadata=None):
        embedding = get_embedding(text)
        self.documents.append({"text": text, "metadata": metadata or {}})
        self.embeddings.append(embedding)
        print("Added;", text[:60] + "...")


    def search(self, query, top_k=3):
        query_embedding = get_embedding(query)


        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, i))
        similarities.sort(reverse = True, key = lambda x: x[0])
        results = []
        for sim, idx in similarities[:top_k]:
            results.append({"similarity": round(sim, 3), "text": self.documents[idx]["text"], "metadata": self.documents[idx]["metadata"]})
        return results
        # create a knowledge base 
print()
print("Building Knowledge Base")
print("="*40)
db = SimpleVectorDB()

db.add("Employees with 1+ year tenure receive 16 weeks paid parental leave.", {"source": "HR Policy", "page": 42})
db.add("vacation acrual: 15 days per year, increasing to 20 after 5 years.", {"source": "HR Policy", "page": 15})
db.add("The office cafeteria serves pizza every Friday.", {"source": "Office Guide", "page": 8})
db.add("Health insurance covers maternity and prenatal care.",{"source": "Benefits Guide", "page": 22})
db.add("Employees with at least one year of service receive 16 weeks of paid parental leave.", {"source": "HR Policy - Page 42"})
db.add("Employees may work remotely up to two days per week with approval from their manager.", {"source": "Remote Work Policy - Page 8"})
db.add("Employees receive 15 days of annual vacation leave each year. This increases to 20 days after five years of service.", {"source": "HR Policy - Page 15"})
db.add("The company provides health insurance that includes maternity care, prenatal services, and general medical treatment.", {"source": "Benefits Guide - Page 22"})
db.add("The company provides annual cybersecurity awareness training covering phishing, password security, social engineering, and data protection.", {"source": "Cybersecurity Policy - Page 14"})
db.add("Employees are expected to attend scheduled meetings and communicate with their teams using approved collaboration platforms.", {"source": "Communication Policy - Page 9"})

#'audio', 'image', 'text', 'video'

# Test semantic search 
print()
print("Search 1: 'How long is ,maternity leave'?")
print("="*40)
results = db.search("How long is maternity leave?", top_k =3)
for r in results:
    print(" [" + str(r['similarity']) + "] " + r['text'])
    print("source:", r['metadata'].get('source'), "page:", r['metadata'].get('page'))
    print()

print("Search 2: 'can i work from home'?")
print("="*40)
results = db.search("Can i work from home?", top_k=3)
for r in results:
    print(" [" + str(r['similarity']) + "] " + r['text'])
    print("source:", r['metadata'].get('source'), "page:", r['metadata'].get('page'))
    print()

print("Search 3: 'what food is available'?")
print("="*40)
results = db.search("what food is available?", top_k=3)
for r in results:
    print(" [" + str(r['similarity']) + "] " + r['text'])
    print("source:", r['metadata'].get('source'), "page:", r['metadata'].get('page'))
    print()

print("Search 3: 'What happens if I want to improve my professional skills?'")
print("="*40)
results = db.search("happens if I want to improve my professional skills?", top_k=3)
for r in results:
    print(" [" + str(r['similarity']) + "] " + r['text'])
    print("source:", r['metadata'].get('source'), "page:", r['metadata'].get('page'))
    print()

print("Search 3: 'What medical support can employees receive during pregnancy?'")
print("="*40)
results = db.search("What medical support can employees receive during pregnancy?", top_k=3)
for r in results:
    print(" [" + str(r['similarity']) + "] " + r['text'])
    print("source:", r['metadata'].get('source'), "page:", r['metadata'].get('page'))
    print()