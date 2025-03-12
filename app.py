import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import os
import time

# Set Streamlit page configuration
st.set_page_config(page_icon='🤖', page_title="Textbook Generator")

# Function to get response from LLaMA2 model (Backend)
def response(input_text, no_words, blog_style):
    # Verify the model path
    model_path = '/home/sohada/Bacbon/TextBook_Generation/Blog-Generation-using-Llama2/model/llama-2-7b-chat.Q2_K.gguf'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return "Model file not found. Please check the path."

    # Initialize the LLaMA2 model
    llm = CTransformers(
        model=model_path,  # Use absolute path
        model_type='llama',
        config={'max_new_tokens': 256, 'temperature': 0.01}
    )

    # Prompt Template
    prompt = PromptTemplate(
        input_variables=['blog_style', 'input_text', 'no_words'],
        template='Write a blog for {blog_style} audience on the topic {input_text} under {no_words} words or less.'
    )

    # Generate the response from LLaMA2
    with st.spinner("Generating blog..."):
        response = llm.invoke(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
        time.sleep(1)  # Simulate a delay (remove this in production)
    st.success("Blog generated successfully!")
    return response

# Streamlit Code (Frontend)
st.header('Generate Blog 🤖')

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Enter number of words')
with col2:
    blog_style = st.selectbox('Writing the blog for ______ audience', ('Beginner', 'Intermediate', 'Pro'))

submit = st.button('Generate')

# Final Response
if submit:
    st.write(response(input_text, no_words, blog_style))

# Test Block to Test the Model Independently
if __name__ == "__main__":
    # Test the model
    print("Testing the model...")
    test_llm = CTransformers(
        model='/home/sohada/Bacbon/TextBook_Generation/Blog-Generation-using-Llama2/model/llama-2-7b-chat.Q2_K.gguf',
        model_type='llama',
        config={'max_new_tokens': 256, 'temperature': 0.01}
    )
    test_response = test_llm.invoke("Hello, world!")
    print("Test Response:", test_response)