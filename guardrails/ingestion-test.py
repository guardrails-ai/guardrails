from guard_rails_api_client import AuthenticatedClient
from guard_rails_api_client.api.default import ingest, get_embeddings
from guard_rails_api_client.models import IngestionPayload

from guardrails.document_store import Document

client = AuthenticatedClient(base_url="http://localhost:5000", follow_redirects=True, token="test-token")

openai_api_key = ""
document = Document("0001", {0: "canine companions say alot about alot"}, { "meta": "test"})
toIngest = {
    'articles': list(document.pages.values()), 
    'metadata': document.metadata, 
    'guardId': document.id
}
'''toIngest['articles'] = list(document.pages.values())
toIngest['metadata'] = document.metadata
toIngest['guardId'] = document.id'''

embeddings = ingest.sync(
            x_openai_api_key=openai_api_key, 
            client=client,
            json_body=IngestionPayload.from_dict(toIngest))
print(embeddings)

'''embeddedItem = get_embeddings.sync(uuid='fa47d370-265e-11ee-8de7-0242ac150005', client=client)
print(embeddedItem)'''