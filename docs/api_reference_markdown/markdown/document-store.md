# Document Store






































































































































, Word documents, etc. A collection of



































































































































arameters:**
* **id** (*str*) – 
* **pages** (*Dict**[**int**,* *str**]*) – 
* **metadata** (*Dict**[**Any**,* *Any**]* *|* *None*) – 
* **Return type:**
None

### *class* guardrails.document_store.DocumentStoreBase

Abstract class for a store that can store text, and metadata from
documents.

The store can be queried by text for similar documents.

#### *abstract* add_document(document)

Adds a document to the store.

* **Parameters:**
**document** ([*Document*](#guardrails.document_store.Document)) – Document object to be added
* **Returns:**
None if the document was added successfully

#### *abstract* add_text(text, meta)

Adds a text to the store.
:param text: Text to add.
:param meta: Metadata to associate with the text.

* **Returns:**
The id of the text.
* **Parameters:**
* **text** (*str*) – 
* **meta** (*Dict**[**Any**,* *Any**]*) – 
* **Return type:**
str

#### *abstract* add_texts(texts)

Adds a list of texts to the store.
:param texts: List of texts to add, and their associalted metadata.
:param example: [{“I am feeling good”, {“sentiment”: “postive”}}]

* **Returns:**
List of ids of the texts.
* **Parameters:**
**texts** (*Dict**[**str**,* *Dict**[**Any**,* *Any**]**]*) – 
* **Return type:**
*List*[str]

#### *abstract* flush()

Flushes the store to disk.

#### *abstract* search(query, k=4)

Searches for pages which contain the text similar to the query.

* **Parameters:**
* **query** (*str*) – Text to search for.
* **k** (*int*) – Number of similar pages to return.
* **Returns:**
List[Pages] List of pages which contains similar texts
* **Return type:**
*List*[[*Page*](#guardrails.document_store.Page)]

### *class* guardrails.document_store.EphemeralDocumentStore

EphemeralDocumentStore is a document store that stores the documents on
local disk and use a ephemeral vector store like Faiss.

#### \_\_init_\_(vector_db, path=None)

Creates a new EphemeralDocumentStore.

* **Parameters:**
* **vector_db** (*VectorDBBase*) – VectorDBBase instance to use for storing the vectors.
* **path** (*str* *|* *None*) – Path to the database file store metadata.

#### add_document(document)

Adds a document to the store.

* **Parameters:**
**document** ([*Document*](#guardrails.document_store.Document)) – Document object to be added
* **Returns:**
None if the document was added successfully

#### add_text(text, meta)

Adds a text to the store.
:param text: Text to add.
:param meta: Metadata to associate with the text.

* **Returns:**
The id of the text.
* **Parameters:**
* **text** (*str*) – 
* **meta** (*Dict**[**Any**,* *Any**]*) – 
* **Return type:**
str

#### add_texts(texts)

Adds a list of texts to the store.
:param texts: List of texts to add, and their associalted metadata.
:param example: [{“I am feeling good”, {“sentiment”: “postive”}}]

* **Returns:**
List of ids of the texts.
* **Parameters:**
**texts** (*Dict**[**str**,* *Dict**[**Any**,* *Any**]**]*) – 
* **Return type:**
*List*[str]

#### flush(path=None)

Flushes the store to disk.

* **Parameters:**
**path** (*str* *|* *None*) – 

#### search(query, k=4)

Searches for pages which contain the text similar to the query.

* **Parameters:**
* **query** (*str*) – Text to search for.
* **k** (*int*) – Number of similar pages to return.
* **Returns:**
List[Pages] List of pages which contains similar texts
* **Return type:**
*List*[[*Page*](#guardrails.document_store.Page)]

### *class* guardrails.document_store.Page

Page holds text and metadata of a page in a document.

It also containts the coordinates of the page in the document.

#### \_\_init_\_(cordinates, text, metadata)

* **Parameters:**
* **cordinates** ([*PageCoordinates*](#guardrails.document_store.PageCoordinates)) – 
* **text** (*str*) – 
* **metadata** (*Dict**[**Any**,* *Any**]*) – 
* **Return type:**
None

### *class* guardrails.document_store.PageCoordinates

PageCoordinates(doc_id, page_num)

#### *static* \_\_new_\_(\_cls, doc_id, page_num)

Create new instance of PageCoordinates(doc_id, page_num)

#### \_asdict()

Return a new dict which maps field names to their values.

#### *classmethod* \_make(iterable)

Make a new PageCoordinates object from a sequence or iterable

#### \_replace(\*\*kwds)

Return a new PageCoordinates object replacing specified fields with new values

#### doc_id

Alias for field number 0

#### page_num

Alias for field number 1

### *class* guardrails.document_store.SqlDocument

#### \_\_init_\_(\*\*kwargs)

A simple constructor that allows initialization from kwargs.

Sets attributes on the constructed instance using the names and
values in `kwargs`.

Only keys that are present as
attributes of the instance’s class are allowed. These could be,
for example, any mapped columns or relationships.
\_init_\_(\*\*kwargs)

> A simple constructor that allows initialization from kwargs.

> Sets attributes on the constructed instance using the names and
> values in `kwargs`.

> Only keys that are present as
> attributes of the instance’s class are allowed. These could be,
> for example, any mapped columns or relationships.
