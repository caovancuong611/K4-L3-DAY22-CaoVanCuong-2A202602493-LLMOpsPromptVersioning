"""
Factory tạo LLM và Embeddings cho 5 providers: openai, gemini, anthropic, ollama, openrouter.

Cách dùng:
    from utils.llm_factory import get_llm, get_embeddings

    llm        = get_llm()            # dùng PROVIDER từ .env
    embeddings = get_embeddings()     # dùng PROVIDER từ .env

    llm_gemini = get_llm("gemini")    # chỉ định provider cụ thể
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def get_llm(provider: str = None, temperature: float = 0.0, for_eval: bool = False):
    """
    Trả về BaseChatModel tương ứng với provider được chọn.

    Args:
        provider    : "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                      Mặc định: đọc PROVIDER từ .env (config.PROVIDER)
        temperature : độ ngẫu nhiên (0.0 = tất định, 1.0 = sáng tạo)
        for_eval    : True → dùng model/key dành cho evaluator (RAGAS judge) nếu có cấu hình
                      (hiện hỗ trợ GEMINI_EVAL_MODEL / GEMINI_EVAL_API_KEY)

    Returns:
        BaseChatModel instance sẵn sàng sử dụng

    Raises:
        ValueError nếu provider không hợp lệ
        ImportError nếu package tương ứng chưa được cài đặt
    """
    provider = (provider or config.PROVIDER).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_EVAL_MODEL if for_eval else config.GEMINI_MODEL,
            google_api_key=config.GEMINI_EVAL_API_KEY if for_eval else config.GOOGLE_API_KEY,
            temperature=temperature,
        )

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.ANTHROPIC_MODEL,
            api_key=config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=temperature,
        )

    elif provider == "openrouter":
        # OpenRouter dùng OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.OPENROUTER_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )


def get_embeddings(provider: str = None):
    """
    Trả về Embeddings instance tương ứng với provider được chọn.

    Lưu ý quan trọng:
        - Anthropic KHÔNG có Embeddings API → tự động fallback về OpenAI embeddings
        - OpenRouter cũng dùng OpenAI embeddings (không có API embeddings riêng)
        - Ollama cần model embedding riêng (mặc định: nomic-embed-text)
          Cài đặt: ollama pull nomic-embed-text

    Args:
        provider: "openai" | "gemini" | "anthropic" | "ollama" | "openrouter"
                  Mặc định: đọc PROVIDER từ .env

    Returns:
        Embeddings instance sẵn sàng sử dụng
    """
    provider = (provider or config.PROVIDER).lower()

    if provider in ("openai", "openrouter"):
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": config.OPENAI_EMBEDDING_MODEL,
            "api_key": config.OPENAI_API_KEY,
        }
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return _RetryCachedEmbeddings(GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        ))

    elif provider == "anthropic":
        # Anthropic không cung cấp Embeddings API → dùng OpenAI thay thế
        print("⚠️  Anthropic không có Embeddings API — đang dùng OpenAI embeddings thay thế.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )

    elif provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    else:
        raise ValueError(
            f"Provider không hợp lệ: '{provider}'. "
            "Chọn một trong: openai, gemini, anthropic, ollama, openrouter"
        )


# ── Embeddings wrapper: retry khi bị rate-limit + cache ra đĩa ────────────
from langchain_core.embeddings import Embeddings as _Embeddings


class _RetryCachedEmbeddings(_Embeddings):
    """
    Bọc một Embeddings instance để:
      - Tự retry (chờ tăng dần) khi API báo 429 / lỗi mạng tạm thời
        → free tier Gemini chỉ cho 100 embed request/phút.
      - Cache vector theo nội dung text (data/embedding_cache.json) để các bước
        sau không gọi lại API cho cùng một câu hỏi / chunk.
    """

    _RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "WinError", "Connection", "timed out", "getaddrinfo")

    def __init__(self, inner, max_retries: int = 8):
        import json
        import threading
        self.inner = inner
        self.model = getattr(inner, "model", type(inner).__name__)
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._path = Path(__file__).parent.parent.parent / "data" / "embedding_cache.json"
        try:
            self._cache = json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            self._cache = {}

    def _key(self, kind: str, text: str) -> str:
        import hashlib
        return hashlib.md5(f"{self.model}|{kind}|{text}".encode("utf-8")).hexdigest()

    def _save(self):
        import json
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._cache), encoding="utf-8")

    def _call(self, fn, *args):
        import time
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args)
            except Exception as e:
                if attempt == self.max_retries or not any(s in str(e) for s in self._RETRYABLE):
                    raise
                wait = min(10 * (attempt + 1), 60)
                print(f"   ⏳ Embedding bị giới hạn/lỗi mạng, thử lại sau {wait}s ...")
                time.sleep(wait)

    def embed_documents(self, texts):
        keys = [self._key("doc", t) for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self._cache]
        if missing:
            vectors = self._call(self.inner.embed_documents, missing)
            with self._lock:
                for t, v in zip(missing, vectors):
                    self._cache[self._key("doc", t)] = v
                self._save()
        return [self._cache[k] for k in keys]

    def embed_query(self, text):
        k = self._key("query", text)
        if k not in self._cache:
            v = self._call(self.inner.embed_query, text)
            with self._lock:
                self._cache[k] = v
                self._save()
        return self._cache[k]
