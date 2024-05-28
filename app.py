import streamlit as st
import fitz  # PyMuPDF
from pymongo import MongoClient
import pandas as pd
import fitz  # PyMuPDF
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import faiss 
from sentence_transformers import SentenceTransformer


# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# define the database URI
client = MongoClient("mongodb+srv://farhan:1234@cluster0.bmlgmq3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

# connect to the database
db = client['mydatabase']
collection = db['mycollection']

# make an instance of the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# functioin to do NLP operations on the resume text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back into a single string
    preprocessed_text = ' '.join(words)
    
    return preprocessed_text


# convert the text to embeddings
def encode_documents(documents):
    embeddings = model.encode(documents)
    return embeddings

# convert the embedding to the Faiss index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance for similarity
    index.add(embeddings)
    return index

# serch the top five resume given the query
def search_index(index, query_embedding, top_k=5): 
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return indices[0]  # Return indices of the most similar resumes


# def create_faiss_index_ip(embeddings):
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dimension)  # Use Inner Product for similarity
#     index.add(embeddings)
#     return index


# Function to read PDF content
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function for Applicants page
def applicants_page():
    st.title("PDF Upload and Skills Assessment")

    st.subheader("Resume")
    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    st.subheader("Personal Information")
    # Name, contact number, and email fields
    name = st.text_input("Name")
    contact_number = st.text_input("Contact Number")
    email = st.text_input("Email")

    # Define weights for each skill
    weights = {
        "PPC Campaign Management": 1,
        "Keyword Research": 0.1,
        "Ad Copywriting": 0.3,
        "Data Analysis": 0.4,
        "Conversion Rate Optimization (CRO)": 0.5,
        "Budget Management": 0.6,
        "Audience Targeting": 0.7,
        "Competitor Analysis": 0.8,
        "Technical Knowledge": 0.9,
    }

    if uploaded_file is not None:
        pdf_text = read_pdf(uploaded_file)

    # Skill levels
    levels = {"Beginner": 3, "Intermediate": 6, "Advanced": 10}

    st.subheader("Skills")

    # Dropdown fields for skills
    skills = {
        "PPC Campaign Management": st.selectbox("PPC Campaign Management", list(levels.keys())),
        "Keyword Research": st.selectbox("Keyword Research", list(levels.keys())),
        "Ad Copywriting": st.selectbox("Ad Copywriting", list(levels.keys())),
        "Data Analysis": st.selectbox("Data Analysis", list(levels.keys())),
        "Conversion Rate Optimization (CRO)": st.selectbox("Conversion Rate Optimization (CRO)", list(levels.keys())),
        "Budget Management": st.selectbox("Budget Management", list(levels.keys())),
        "Audience Targeting": st.selectbox("Audience Targeting", list(levels.keys())),
        "Competitor Analysis": st.selectbox("Competitor Analysis", list(levels.keys())),
        "Technical Knowledge": st.selectbox("Technical Knowledge", list(levels.keys())),
    }
    
    # Submit button
    if st.button("Submit"):
        if uploaded_file is not None:
            st.success("Form submitted successfully!")
            skills_scores = {skill: levels[level] * weights[skill] for skill, level in skills.items()}
            average_score = sum(skills_scores.values()) / len(skills_scores)

            form_data = {
                "Name": name,
                "Contact Number": contact_number,
                "Email": email,
                "Skills": skills_scores,
                "Average Skill Score": average_score,
                "Resume": pdf_text
            }

            # add the form data to the database
            collection.insert_one(form_data)

        else:
            st.error("Please upload a PDF file.")

# Function for Admin page
def admin_page():
    st.title("Admin Page")

    st.subheader("Clear Database")
    if st.button("Clear Database"):
        collection.delete_many({})
        st.success("All data in the database has been deleted.")
    
    st.subheader("Add Job Description and Run Algorithm")
    job_description = st.text_area("Enter job description here")
    if st.button("Run Algorithm"):
        if job_description:
            documents = list(collection.find())
            if documents:
                # Convert documents to DataFrame
                df = pd.DataFrame(documents)
                df.sort_values(by="Average Skill Score", ascending = False, inplace = True)
                df = df.head(5)
                df['processed_resume'] = df['Resume'].apply(preprocess_text)


                preprocessed_resumes = df['processed_resume'].tolist()
                preprocessed_job_description = preprocess_text(job_description)

                # 2. Semantic Encoding
                resume_embeddings = encode_documents(preprocessed_resumes)
                job_description_embedding = encode_documents([preprocessed_job_description])[0]

                # 3. FAISS Index and Search
                index = create_faiss_index(resume_embeddings)
                top_resume_indices = search_index(index, job_description_embedding, top_k=3) # Get top 2

                # 4. Output Results
                print("Top Matched Resumes:")
                for i in top_resume_indices:
                    st.write(f"Resume # {i+1}")
                    st.write(df["Resume"].iloc[i]) 
                

            else:
                st.write("The database is empty.")
        else:
            st.write("Enter Job Description first")

# Main function
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Applicants", "Admin"])

    if page == "Applicants":
        applicants_page()
    elif page == "Admin":
        admin_page()

if __name__ == "__main__":
    main()
