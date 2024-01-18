import os  
import json  
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv  
from tenacity import retry, wait_random_exponential, stop_after_attempt  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryLanguage,
    QueryCaptionType,
    RawVectorQuery,
    VectorizableTextQuery,
    VectorFilterMode 
)
from azure.search.documents.indexes.models import (  
    AzureOpenAIEmbeddingSkill,  
    AzureOpenAIParameters,  
    AzureOpenAIVectorizer,  
    ExhaustiveKnnParameters,  
    ExhaustiveKnnVectorSearchAlgorithmConfiguration,
    FieldMapping,  
    HnswParameters,  
    HnswVectorSearchAlgorithmConfiguration,  
    IndexProjectionMode,  
    InputFieldMappingEntry,  
    OutputFieldMappingEntry,  
    PrioritizedFields,    
    SearchField,  
    SearchFieldDataType,  
    SearchIndex,  
    SearchIndexer,  
    SearchIndexerDataContainer,  
    SearchIndexerDataSourceConnection,  
    SearchIndexerIndexProjectionSelector,  
    SearchIndexerIndexProjections,  
    SearchIndexerIndexProjectionsParameters,  
    SearchIndexerSkillset,  
    SemanticConfiguration,  
    SemanticField,  
    SemanticSettings,  
    SplitSkill,  
    VectorSearch,  
    VectorSearchAlgorithmKind,  
    VectorSearchAlgorithmMetric,  
    VectorSearchProfile,  
)  

from azure.storage.blob import BlobServiceClient
  
# Configure environment variables  
load_dotenv("claim-processing autogen\hr_agent\.env")  

service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT") 
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME") 
key = os.getenv("AZURE_SEARCH_ADMIN_KEY") 
model: str = "text-embedding-ada-002" 
blob_connection_string = os.getenv("BLOB_CONNECTION_STRING")  
container_name = os.getenv("BLOB_CONTAINER_NAME")  
credential = AzureKeyCredential(key)

# Retrieve the documents from blob storage
blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
container_client = blob_service_client.get_container_client(container_name)
blobs = container_client.list_blobs()

first_blob = next(blobs)
blob_url = container_client.get_blob_client(first_blob).url
print(f"URL of the first blob: {blob_url}")

## Connect to the data source
ds_client = SearchIndexerClient(service_endpoint, AzureKeyCredential(key))
container = SearchIndexerDataContainer(name=container_name)
data_source_connection = SearchIndexerDataSourceConnection(
    name=f"{index_name}-blob",
    type="azureblob",
    connection_string=blob_connection_string,
    container=container
)
data_source = ds_client.create_or_update_data_source_connection(data_source_connection)

print(f"Data source '{data_source.name}' created or updated")

# Create a search index  
index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)  
fields = [  
    SearchField(name="parent_id", type=SearchFieldDataType.String, sortable=True, filterable=True, facetable=False),  
    SearchField(name="title", type=SearchFieldDataType.String),  
    SearchField(name="chunk_id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True, analyzer_name="keyword"),  
    SearchField(name="chunk", type=SearchFieldDataType.String, sortable=False, filterable=False, facetable=False),  
    SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), vector_search_dimensions=1536, vector_search_profile="myHnswProfile"),  
]  
  
# Configure the vector search configuration  
vector_search = VectorSearch(  
    algorithms=[  
        HnswVectorSearchAlgorithmConfiguration(  
            name="myHnsw",  
            kind=VectorSearchAlgorithmKind.HNSW,  
            parameters=HnswParameters(  
                m=4,  
                ef_construction=400,  
                ef_search=500,  
                metric=VectorSearchAlgorithmMetric.COSINE,  
            ),  
        ),  
        ExhaustiveKnnVectorSearchAlgorithmConfiguration(  
            name="myExhaustiveKnn",  
            kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,  
            parameters=ExhaustiveKnnParameters(  
                metric=VectorSearchAlgorithmMetric.COSINE,  
            ),  
        ),  
    ],  
    profiles=[  
        VectorSearchProfile(  
            name="myHnswProfile",  
            algorithm="myHnsw",  
            vectorizer="myOpenAI",  
        ),  
        VectorSearchProfile(  
            name="myExhaustiveKnnProfile",  
            algorithm="myExhaustiveKnn",  
            vectorizer="myOpenAI",  
        ),  
    ],  
    vectorizers=[  
        AzureOpenAIVectorizer(  
            name="myOpenAI",  
            kind="azureOpenAI",  
            azure_open_ai_parameters=AzureOpenAIParameters(  
                resource_uri=os.getenv("OPENAI_API_BASE"),  
                deployment_id=model,  
                api_key=os.getenv("OPENAI_API_KEY"),  
            ),  
        ),  
    ],  
)  
  
semantic_config = SemanticConfiguration(  
    name="my-semantic-config",  
    prioritized_fields=PrioritizedFields(  
        prioritized_content_fields=[SemanticField(field_name="chunk")]  
    ),  
)  
  
# Create the semantic settings with the configuration  
semantic_settings = SemanticSettings(configurations=[semantic_config])  
  
# Create the search index with the semantic settings  
index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_settings=semantic_settings)  
result = index_client.create_or_update_index(index)  
print(f"{result.name} created")  

# Create OpenAI Embedding Skill and Integrated vectorization skills and add it to Skillset
# Create a skillset  
skillset_name = f"{index_name}-skillset"  
  
split_skill = SplitSkill(  
    description="Split skill to chunk documents",  
    text_split_mode="pages",  
    context="/document",  
    maximum_page_length=1000,  
    page_overlap_length=200,  
    inputs=[  
        InputFieldMappingEntry(name="text", source="/document/content"),  
    ],  
    outputs=[  
        OutputFieldMappingEntry(name="textItems", target_name="pages")  
    ],  
)  
  
embedding_skill = AzureOpenAIEmbeddingSkill(  
    description="Skill to generate embeddings via Azure OpenAI",  
    context="/document/pages/*",  
    resource_uri=os.getenv("OPENAI_API_BASE"),  
    deployment_id=model,  
    api_key=os.getenv("OPENAI_API_KEY"),  
    inputs=[  
        InputFieldMappingEntry(name="text", source="/document/pages/*"),  
    ],  
    outputs=[  
        OutputFieldMappingEntry(name="embedding", target_name="vector")  
    ],  
)  
  
index_projections = SearchIndexerIndexProjections(  
    selectors=[  
        SearchIndexerIndexProjectionSelector(  
            target_index_name=index_name,  
            parent_key_field_name="parent_id",  
            source_context="/document/pages/*",  
            mappings=[  
                InputFieldMappingEntry(name="chunk", source="/document/pages/*"),  
                InputFieldMappingEntry(name="vector", source="/document/pages/*/vector"),  
                InputFieldMappingEntry(name="title", source="/document/metadata_storage_name"),  
            ],  
        ),  
    ],  
    parameters=SearchIndexerIndexProjectionsParameters(  
        projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS  
    ),  
)  
  
skillset = SearchIndexerSkillset(  
    name=skillset_name,  
    description="Skillset to chunk documents and generating embeddings",  
    skills=[split_skill, embedding_skill],  
    index_projections=index_projections,  
)  
  
client = SearchIndexerClient(service_endpoint, AzureKeyCredential(key))  
client.create_or_update_skillset(skillset)  
print(f"{skillset.name} created")  

## Create an indexer 
indexer_name = f"{index_name}-indexer"  
  
indexer = SearchIndexer(  
    name=indexer_name,  
    description="Indexer to index documents and generate embeddings",  
    skillset_name=skillset_name,  
    target_index_name=index_name,  
    data_source_name=data_source.name,  
    # Map the metadata_storage_name field to the title field in the index to display the PDF title in the search results  
    field_mappings=[FieldMapping(source_field_name="metadata_storage_name", target_field_name="title")]  
)  
  
indexer_client = SearchIndexerClient(service_endpoint, AzureKeyCredential(key))  
indexer_result = indexer_client.create_or_update_indexer(indexer)  
  
# Run the indexer  
indexer_client.run_indexer(indexer_name)  
print(f' {indexer_name} created')  

query = "Are Laptops eligible for reimbursement?"  
  
search_client = SearchClient(service_endpoint, index_name, credential=credential)
vector_query = VectorizableTextQuery(text=query, k=3, fields="vector", exhaustive=True)
# Use the below query to pass in the raw vector query instead of the query vectorization
# vector_query = RawVectorQuery(vector=generate_embeddings(query), k=3, fields="vector")
  
results = search_client.search(  
    search_text=query,  
    vector_queries= [vector_query],
    select=["parent_id", "chunk_id", "chunk"],
    top=3
)  
  
for result in results:  
    # print(f"parent_id: {result['parent_id']}")  
    # print(f"chunk_id: {result['chunk_id']}")  
    # print(f"Score: {result['@search.score']}")  
    print(f"Content: {result['chunk']}")   
