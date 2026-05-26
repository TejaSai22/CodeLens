from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class IndexRequest(BaseModel):
    repo_input: str = Field(..., description="GitHub URL or local path to repository")

class IndexStartedResponse(BaseModel):
    success: bool
    message: str
    repo_id: str
    display_name: str
    status: str

class QueryRequest(BaseModel):
    repo_id: str = Field(..., description="Id of the indexed repository to query")
    query: str = Field(..., description="User question about the codebase")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Previous chat messages")
    language_filter: Optional[str] = Field(default=None, description="Optional language filter, e.g. 'python'")

class SourceChunk(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    similarity: float
    chunk_type: str
    name: str
    content: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]

class RepoInfo(BaseModel):
    repo_id: str
    display_name: str
    source: str
    status: str
    files_indexed: int
    chunks_created: int
    progress: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str

class DeleteResponse(BaseModel):
    success: bool
    repo_id: str
