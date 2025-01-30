import os
from constants import GROQ_API_KEY as key
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts  import PromptTemplate, FewShotPromptTemplate

#setup key
os.environ['GROQ_API_KEY']=key

# Streamlit UI
st.set_page_config(page_title="Findlation - AI Dictionary", page_icon="📖", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        text-align: center;
        color: #ff6f61;
    }
    .stTextInput {
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #ff6f61;
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 10px;
        width: 100%;
    }
    .stMarkdown {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# App Title
st.image("D:\practice_AI\langchain\word_finder\pic.jpeg", 
         width=100)
st.title("📖 Findlation - AI Dictionary")

#llm
llm=ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.8
)

#user input
word = st.text_input("Enter a word:", placeholder="e.g., Serendipity")
target_lang = st.text_input("Translate to (Language):", placeholder="e.g., Spanish")

#prompt1
if st.button("Find Meaning & Translate"):
    if word:
        #translation prompt
        prompt1=PromptTemplate.from_template("Translate {word} into {target_lang}")
        translation=prompt1.format(word=word, target_lang=target_lang)
        translation_response=llm.invoke(translation)
        translation_text = translation_response.content.strip() #to only gate the content
        
        #for meaning,synonym and antonymm
        #example
        examples=[
            {
        "word": "profile", 
        "meaning":"1.an outline of something, especially a person's face, as seen from one side.\n 2.a short article giving a description of a person or organization.",
        "antonyms":"1.obscurity 2.opacity",
        "synonyms":"1.outline 2.description\n"
        },
        {
        "word": "fresh", 
        "meaning":"1.(of food) recently made or obtained; not tinned, frozen, or otherwise preserved..\n 2.not previously known or used; new or different.",
        "antonyms":"1.stale 2.old",
        "synonyms":"1.garden-fresh 2.new\n"
        }
        ]
        #prompt2
        format= """
        Word: {word}
        Meaning: {meaning}
        Antonyms: {antonyms}
        Synonyms: {synonyms}"""
        
        prompt2=PromptTemplate(
            input_variables=["word","meaning", "antonyms", "synonyms"],
            template=format
        )
        
        few_shot_prompt=FewShotPromptTemplate(
            examples=examples, #example
            example_prompt=prompt2, #this is how we want to format the examples when we instert them into the prompt
            prefix="Strictly give all the possible meanings and 5 antonyms, synonyms of the given input:\n", #text that goes before the examples in the prompt, usually instructions
            suffix="Word: {word}\n Meaning: \nAntonyms: \nSynonyms: \n", #text that goes after the examples in the prompt, usually the user input
            input_variables=["word"], #input the overall prompt expects
            example_separator="\n", #It will join the prefix, examples and suffix
        )
        
        des=few_shot_prompt.format(word=word)
        response=llm.invoke(des)
        definition_text = response.content.strip()#to only gate the content
        
        # Display Outputs
        st.success(f"**Translation:** {translation_text}")
        st.markdown(f"### 📌 Definition of '{word}'")
        st.markdown(f"```{definition_text}```")
        
    else:
        st.warning("⚠️ Please enter a word to get its meaning and translation.")