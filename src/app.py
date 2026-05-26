import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/api/v1/rag"

# Page configuration
st.set_page_config(
    page_title="CodeLens",
    page_icon="🔍",
    layout="wide"
)

def index_repository(repo_input: str, input_type: str):
    """Index a repository via API."""
    try:
        with st.spinner("Indexing repository... This may take a while depending on size."):
            response = requests.post(
                f"{API_URL}/index",
                json={"repo_input": repo_input}
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ {data['message']}")
                return True
            else:
                error_msg = response.json().get('detail', 'Unknown error')
                st.error(f"Error indexing repository: {error_msg}")
                return False
                
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Is FastAPI running on port 8000?")
        return False
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def handle_query(query: str):
    """Query codebase via API."""
    try:
        # Prepare context
        history = st.session_state.get('messages', [])
        recent_history = history[-4:] if len(history) > 4 else history
        
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": query,
                "conversation_history": recent_history
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data['answer']
            sources = data['sources']
            
            # Store in session
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Display answer
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # Show sources in expander if any
                if sources:
                    with st.expander(f"📄 Sources ({len(sources)})"):
                        for i, chunk in enumerate(sources, 1):
                            file_path = chunk.get('file_path', 'unknown')
                            start_line = chunk.get('start_line', 0)
                            end_line = chunk.get('end_line', 0)
                            similarity = chunk.get('similarity', 0.0)
                            chunk_type = chunk.get('chunk_type', 'code')
                            name = chunk.get('name', '')
                            
                            st.markdown(f"**{i}. {file_path}** (lines {start_line}-{end_line})")
                            st.markdown(f"Relevance: {similarity:.2%} | Type: {chunk_type}")
                            if name:
                                st.markdown(f"Name: `{name}`")
                            st.code(chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'])
                            st.markdown("---")
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            st.error(f"API Error: {error_msg}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend API. Is FastAPI running on port 8000?")
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def main():
    """Main application."""

    # Title
    st.title("🔍 CodeLens")
    st.markdown("*RAG-based Codebase Q&A System*")
    
    # API Status check
    api_online = False
    try:
        health = requests.get("http://localhost:8000/health", timeout=2)
        if health.status_code == 200:
            api_online = True
            indexed_chunks = health.json().get('indexed_chunks', 0)
    except:
        indexed_chunks = 0
        
    if not api_online:
        st.error("Backend API is offline. Please run `uvicorn src.api.main:app --reload` in another terminal.")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'repo_indexed' not in st.session_state:
        st.session_state.repo_indexed = (indexed_chunks > 0)

    # Sidebar
    with st.sidebar:
        st.header("Repository Configuration")

        input_type = st.radio(
            "Select input type:",
            ["GitHub URL", "Local Path"]
        )

        if input_type == "GitHub URL":
            repo_input = st.text_input(
                "GitHub Repository URL:",
                placeholder="https://github.com/username/repo"
            )
        else:
            repo_input = st.text_input(
                "Local Path:",
                placeholder="/path/to/local/repo"
            )

        if st.button("Index Repository", type="primary", disabled=not api_online):
            if repo_input:
                success = index_repository(repo_input, input_type)
                if success:
                    st.session_state.repo_indexed = True
                    st.session_state.messages = []  # Clear chat history
            else:
                st.warning("Please provide a repository URL or path")

        st.divider()
        if st.session_state.repo_indexed:
            st.success(f"✅ Ready (Backend chunks: {indexed_chunks})")
        else:
            st.info("ℹ️ No repository indexed")

    # Main chat interface
    st.header("Chat Interface")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask a question about the codebase...", disabled=not api_online or not st.session_state.repo_indexed)
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        handle_query(prompt)


if __name__ == "__main__":
    main()
