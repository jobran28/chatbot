import streamlit as st
import pandas as pd
from chatbot import (
    remove_stop_words,
    vectorize_questions,
    chatbot_response,
    load_dataset
)

def initialize_chatbot(dataset):
    # Preprocess the dataset by removing stop words
    dataset['Question'] = dataset['Question'].apply(remove_stop_words)

    # Build vocabulary from dataset questions
    vocabulary = set(' '.join(dataset['Question'].tolist()).split())

    # Generate TF-IDF vectors for dataset questions
    vectorizer, question_vectors = vectorize_questions(dataset['Question'].tolist())
    
    return vocabulary, vectorizer, question_vectors

def main():
    st.set_page_config(
        page_title="Greece FAQ Bot",
        page_icon="🇬🇷",
        layout="centered"
    )

    st.title("🇬🇷 Greece FAQ Bot")
    
    # File uploader in sidebar
    st.sidebar.header("Upload Knowledge Base")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file (with 'Question' and 'Answer' columns)",
        type=['csv']
    )

    # Initialize session state for messages if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if uploaded_file is not None:
        try:
            dataset = pd.read_csv(uploaded_file)
            
            # Validate CSV structure
            if 'Question' not in dataset.columns or 'Answer' not in dataset.columns:
                st.error("The CSV file must contain 'Question' and 'Answer' columns!")
                return

            # Initialize chatbot components if not already done
            if 'chatbot_initialized' not in st.session_state:
                with st.spinner("Initializing chatbot..."):
                    vocabulary, vectorizer, question_vectors = initialize_chatbot(dataset)
                    st.session_state.vocabulary = vocabulary
                    st.session_state.vectorizer = vectorizer
                    st.session_state.question_vectors = question_vectors
                    st.session_state.dataset = dataset
                    st.session_state.chatbot_initialized = True
                st.success("Chatbot initialized successfully! You can now start chatting.")

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("What would you like to ask?"):
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Generate and display assistant response
                with st.chat_message("assistant"):
                    response = chatbot_response(
                        prompt,
                        st.session_state.vocabulary,
                        st.session_state.vectorizer,
                        st.session_state.question_vectors,
                        st.session_state.dataset
                    )
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    else:
        st.info("👈 Please upload a CSV file with your Q&A knowledge base to start chatting!")

if __name__ == "__main__":
    main()
