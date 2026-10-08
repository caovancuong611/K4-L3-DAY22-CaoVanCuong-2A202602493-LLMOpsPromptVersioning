"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    import hashlib
    import time
    from langchain_community.vectorstores import FAISS

    # Cache index theo (model embedding + nội dung chunks) để các bước sau không phải embed lại
    # — tránh vượt quota free tier (vd. Gemini: 100 embed request/phút).
    model_name = getattr(embeddings, "model", type(embeddings).__name__)
    key = hashlib.md5((str(model_name) + "\x00" + "\x00".join(chunks)).encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(__file__).parent.parent.parent / "data" / "faiss_cache" / key
    if (cache_dir / "index.faiss").exists():
        print(f"📦 Dùng FAISS index đã cache ({len(chunks)} chunks)")
        return FAISS.load_local(str(cache_dir), embeddings, allow_dangerous_deserialization=True)

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")
    batch_size = 80
    vectorstore = None
    for start in range(0, len(chunks), batch_size):
        if start:
            time.sleep(61)  # chờ reset quota/phút
        batch = chunks[start:start + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_texts(batch, embeddings)
        else:
            vectorstore.add_texts(batch)
        print(f"   ... đã embed {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")
    vectorstore.save_local(str(cache_dir))
    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore
