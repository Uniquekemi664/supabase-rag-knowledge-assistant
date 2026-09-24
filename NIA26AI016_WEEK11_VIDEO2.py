from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / '.env')

#Connecting to supabase

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY or not GROQ_API_KEY:
	raise RuntimeError(
		'Missing SUPABASE_URL, SUPABASE_SERVICE_KEY, or GROQ_API_KEY. '
		'Add them to a .env file next to this script.'
	)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

#Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#Load the groq client 
groq_client = Groq(api_key=GROQ_API_KEY)

print("Connected to Supabase")
print("Embedding Model loaded")
print('Groq Client Ready')

#Store Document with embedding 

def store_document(title, content, source, page_number=1): 
	#Generate embedding locally
	embedding = embedding_model.encode(content).tolist()

	#Insert into supabase 
	result = supabase.table('documents').insert({'title':title, 'content':content, 'source':source, 'page_number':page_number, 'embedding':embedding}).execute()
	print("Stored:", title)
	return result.data

#Search for similar document 

def search_similar(query, top_k=5, threshold=0.3): #searching for document similar to the query

	#embed the query
	query_embedding = embedding_model.encode(query).tolist()

	#Call the match_documents function in supabase 
	result = supabase.rpc('match_documents', {'query_embedding': query_embedding, 'match_threshold':threshold, 'match_count':top_k}).execute()
	return result.data

#Complete RAG Query

def rag_query(question):
	print()
	print('Question:', question)
	print("="*40)

	#Search for relevant document 
	docs = search_similar(question)
	context = ""

	if not docs:
		print("No relevant documents found.")
		return "I couldn't find relevant information in our documents."
	print("Found", len(docs), "relevant documents:")
	for d in docs:
		print("-", d['title'], "(similarity:" + str(round(d['similarity'], 3)) + ")")

	#Build context ffrom retrieved documents 
	for d in docs:
		context += "[" + d['title'] + ", page" + str(d["page_number"]) + "]\n"
		context += d['content'] + "\n\n"

	# Generate answer using groq
	response = groq_client.chat.completions.create(
		model="openai/gpt-oss-120b",
		messages=[
			{"role": "system", "content": "Answer the question based ONLY on the provided documents. Cite sources like[Title, page x]. If the documents don't contain the answer, say so."},
			{"role": "user", "content": "DOCUMENTS:\n" + context + "\nQUESTION:" + question}
		],
		temperature=0.2
	)
	answer = response.choices[0].message.content

	print()
	print("ANSWER:")
	print(answer)
	return(answer)

# Test it 
# Store sample Documents
print("Storing documents")
print("="*40)
store_document("Employee Handbook", "Employees with one year or more tenure recieve sixteen weeks of paid parental leave. This applies to both birth and adoption.", "handbook.pdf", 42)
store_document("IT Policy", "Passwords must be changed every 90 days. Two-factor authentication is required for all company accounts.", "it_policy.pdf", 15)
store_document("Remote Work Policy", "Remote work is permitted up to three days per week after the probation period. Full remote requires manager approval.", "remote_policy.pdf", 3)
store_document("Benefit Guide", "Health insurance covers maternity and prenatal care, Dental and vision plans are available as add-ons.", "benfits.pdf", 22)
store_document("Office Guide", "The office cafeteria serves hot lunch free Monday to Friday. Pizza Friiday is a company tradition.", "office_guide.pdf", 8)
store_document("Leave Policy","Employees with at least one year of service receive 16 weeks of paid parental leave.", "Office_handbook.pdf", 42)
store_document("Remote Work Policy", "Employees may work remotely up to two days per week with approval from their manager.", "Remote_work_policy.pdf", 8)
store_document("Employee Handbook", "Employees receive 15 days of annual vacation leave each year. This increases to 20 days after five years of service.", "HR_policy.pdf",  15)
store_document("Health Policy", "The company provides health insurance that includes maternity care, prenatal services, and general medical treatment.","Benefits_guide.pdf", 22)
store_document("Employee Training Policy", "The company provides annual cybersecurity awareness training covering phishing, password security, social engineering, and data protection.", "Cybersecurity_policy.pdf", 14)
store_document("Employee Meetings Policy", "Employees are expected to attend scheduled meetings and communicate with their teams using approved collaboration platforms.", "Communication_policy.pdf", 9)

print()
print("="*50)
print("Test 1: HR Question")
print("="*50)
rag_query("What is the parental leave policy?")

print()
print()
print("="*50)
print("Test 2: IT Question")
print("="*50)
rag_query("How often do i need to change my password?")

print()
print()
print("="*50)
print("Test 3: Question with no answer in the documents")
print("="*50)
rag_query("What is the company policy on stock options")

print()
print()
print("="*50)
print("Test 4: Lunch Question")
print("="*50)
rag_query("What is the company lunch period")

print()
print()
print("="*50)
print("Test 3: Health Question")
print("="*50)
rag_query("What is the company policy for sick leave")

print()
print()
print("="*50)
print("Test 3: Training Question")
print("="*50)
rag_query("Are all certificate training free for all staffs")

print()
print()
print("="*50)
print("Test 3: Meeting Question")
print("="*50)
rag_query("Can one join meetings online if on the field")

print()
print()
print("="*50)
print("Test 3: Insurance Question")
print("="*50)
rag_query("Can health insurance be made optional")

print()
print()
print("="*50)
print("Test 3: Remote work Question")
print("="*50)
rag_query("Can i sign on later when working online if network fails")

print()
print()
print("="*50)
print("Test 3: IT Question")
print("="*50)
rag_query("How do i reset my password if i forgot it")
