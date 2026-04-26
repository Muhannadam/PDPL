from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    # Paths
    project_root: Path = Path(__file__).resolve().parent
    data_folder: str = "data"
    markdown_folder: str = "markdown"
    vectorstore_folder: str = "vectorstore"
    faiss_folder: str = "faiss_index"

    # Models — يجب أن تبقى مطابقة لكود Colab
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    groq_model: str = "llama-3.3-70b-versatile"

    # Retrieval — يجب أن تبقى مطابقة لكود Colab
    retrieval_k: int = 7
    final_max_sources: int = 4

    # Generation
    temperature: float = 0.0

    @property
    def vectorstore_path(self) -> Path:
        return (
            self.project_root
            / self.data_folder
            / self.vectorstore_folder
            / self.faiss_folder
        )

    @property
    def markdown_path(self) -> Path:
        return (
            self.project_root
            / self.data_folder
            / self.markdown_folder
        )
