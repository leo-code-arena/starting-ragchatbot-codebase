import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    
    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    def validate(self) -> Optional[str]:
        """Return a human-readable problem with this config, or None if it's usable.

        Catches the common setup mistake of copying .env.example without editing
        it, which otherwise only shows up as a 401 on every chat query.
        """
        key = self.ANTHROPIC_API_KEY.strip()
        if not key:
            return ("ANTHROPIC_API_KEY is not set. Create a .env file in the repo root "
                    "with ANTHROPIC_API_KEY=<your key> (see .env.example).")
        if key == "your-anthropic-api-key-here":
            return ("ANTHROPIC_API_KEY is still the placeholder from .env.example. "
                    "Replace it in .env with your real Anthropic API key.")
        return None

config = Config()


