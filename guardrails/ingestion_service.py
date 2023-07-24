import os
import hashlib
from guardrails.document_store import Document, DocumentStoreBase
from guard_rails_api_client import AuthenticatedClient
from typing import Any, Dict, List, Optional
from guard_rails_api_client.api.default import ingest
from guard_rails_api_client.models import IngestionPayload

try:
    import sqlalchemy
    import sqlalchemy.orm as orm
except ImportError:
    sqlalchemy = None
    orm = None


class IngestionServiceDocumentStore(DocumentStoreBase):
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = (
            base_url
            if base_url is not None
            else os.environ.get("GUARDRAILS_BASE_URL", "http://localhost:5000") # TODO: switch the default from localhost to our hosted endpoint
        )
        self.api_key = (
            api_key if api_key is not None else os.environ.get("GUARDRAILS_API_KEY")
        )
        self._client = AuthenticatedClient(
            base_url=self.base_url,
            follow_redirects=True,
            token=self.api_key,
            timeout=300,
        )
    
    def add_document(self, document: Document, openai_api_key: Optional[str] = None):
        openai_api_key = (
            openai_api_key
            if openai_api_key is not None
            else os.environ.get("OPENAI_API_KEY")
        )
        toIngest = {
            'articles': list(document.pages.values()), 
            'metadata': document.metadata, 
            'guardId': document.id
        }
        return ingest.sync(
            x_openai_api_key=openai_api_key, 
            client=self._client,
            json_body=IngestionPayload.from_dict(toIngest))

    def add_text(self, text: str, meta: Dict[Any, Any], openai_api_key: Optional[str] = None): 
        openai_api_key = (
            openai_api_key
            if openai_api_key is not None
            else os.environ.get("OPENAI_API_KEY")
        )
        hash = hashlib.md5()
        id = hash.hexdigest()

        toIngest = {
            'articles': [text], 
            'metadata': meta, 
            'guardId': id #TODO: pass in the actual guardId
        }

        ingested = ingest.sync(
            x_openai_api_key=openai_api_key, 
            client=self._client,
            json_body=IngestionPayload.from_dict(toIngest))
        
        return ingested
    
    def add_texts(self, texts: Dict[str, Dict[Any, Any]], openai_api_key: Optional[str] = None):
        openai_api_key = (
            openai_api_key
            if openai_api_key is not None
            else os.environ.get("OPENAI_API_KEY")
        )
        ingested_items = []
        for text, meta in texts.items():
            ingested = self.add_text(text, meta, openai_api_key)
            ingested_items.append(ingested)
        return ingested_items
    
    def flush(self, path: Optional[str] = None):
        raise NotImplementedError 
    
    def search(self, query: str, k: int = 4):
        raise NotImplementedError