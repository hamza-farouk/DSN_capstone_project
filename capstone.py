import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Margie's Travel Assistant",
    page_icon="✈️",
    layout="wide"
)

@st.cache_resource
def initialize_azure_client():
    """Initialize Azure OpenAI client - cached to avoid recreating"""
    try:
        open_ai_endpoint = os.getenv("OPEN_AI_ENDPOINT")
        open_ai_key = os.getenv("OPEN_AI_KEY")
        
        if not open_ai_endpoint or not open_ai_key:
            st.error("Missing Azure OpenAI configuration. Please check your .env file.")
            return None
            
        client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=open_ai_endpoint,
            api_key=open_ai_key
        )
        return client
    except Exception as e:
        st.error(f"Failed to initialize Azure client: {str(e)}")
        return None

def get_rag_response(chat_client, messages):
    """Get response from Azure RAG system"""
    try:
        # Get configuration
        chat_model = os.getenv("CHAT_MODEL")
        embedding_model = os.getenv("EMBEDDING_MODEL")
        search_url = os.getenv("SEARCH_ENDPOINT")
        search_key = os.getenv("SEARCH_KEY")
        index_name = os.getenv("INDEX_NAME")
        
        # RAG parameters for Azure AI Search
        rag_params = {
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": {
                        "endpoint": search_url,
                        "index_name": index_name,
                        "authentication": {
                            "type": "api_key",
                            "key": search_key,
                        },
                        "query_type": "vector",
                        "embedding_dependency": {
                            "type": "deployment_name",
                            "deployment_name": embedding_model,
                        },
                    }
                }
            ],
        }
        
        # Get response from Azure OpenAI with RAG
        response = chat_client.chat.completions.create(
            model=chat_model,
            messages=messages,
            extra_body=rag_params
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "you are an expert on my history and the projects i have worked on."}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# App header
st.title("My Project Assistant")
st.markdown("*Your AI-powered project companion powered by Azure RAG*")

# Initialize Azure client
chat_client = initialize_azure_client()

if chat_client is None:
    st.stop()

# Sidebar with controls
with st.sidebar:
    st.header("Chat Controls")
    
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.messages = [
            {"role": "system", "content": "You are a travel assistant that provides information on travel services available from Margie's Travel."}
        ]
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This chatbot uses Azure RAG to provide information about the project I have worked on knowledge base. Download the project below")
    
    # Display environment status
    st.markdown("---")
    st.markdown("### Configuration Status")
    config_items = [
        ("Azure OpenAI Endpoint", os.getenv("OPEN_AI_ENDPOINT")),
        ("Chat Model", os.getenv("CHAT_MODEL")),
        ("Embedding Model", os.getenv("EMBEDDING_MODEL")),
        ("Search Endpoint", os.getenv("SEARCH_ENDPOINT")),
        ("Index Name", os.getenv("INDEX_NAME"))
    ]
    
    for item, value in config_items:
        if value:
            st.success(f"✅ {item}")
        else:
            st.error(f"❌ {item}")

# Main chat interface
st.markdown("---")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about my projects..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Add user message to session messages for API
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching project information..."):
            response = get_rag_response(chat_client, st.session_state.messages)
        
        st.markdown(response)
        
        # Add assistant response to chat history and session messages
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Powered by Azure OpenAI and Azure AI Search | Built with Streamlit"
    "</div>", 
    unsafe_allow_html=True
)

# Display some example prompts if chat is empty
if len(st.session_state.chat_history) == 0:
    st.markdown("### 💡 Try asking about:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.link_button("📚 Download My Project", url="https://drive.google.com/file/d/1LRef3AGb0MA95blw5Y6uI5MSAjlFOUi_/view?usp=drive_link"):     
              st.rerun()
    
