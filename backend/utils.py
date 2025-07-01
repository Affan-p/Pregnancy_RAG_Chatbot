import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import (SimpleDirectoryReader,Document, VectorStoreIndex, StorageContext, load_index_from_storage)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import CSVReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq



load_dotenv()


embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Groq(
    model="llama-3.1-8b-instant",  
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=500,
    temperature=0.1
)


Settings.embed_model = embed_model
Settings.llm = llm


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")

def get_vector_store():
    
    pinecone_index = pc.Index(index_name)
    return PineconeVectorStore(pinecone_index=pinecone_index)

def get_storage_context(for_rebuild=False):
    
    vector_store = get_vector_store()
    persist_dir = "./storage"
    
    if for_rebuild or not os.path.exists(persist_dir):
    
        return StorageContext.from_defaults(vector_store=vector_store)
    else:
    
        return StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=persist_dir
        )





def get_and_chunk_documents():

    try:

        file_extractor = {".csv": CSVReader()}


        documents = SimpleDirectoryReader(
            "../knowledge_base", 
            file_extractor=file_extractor
        ).load_data()

        print(f"📖 Loaded {len(documents)} documents")

        node_parser = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=95, 
            embed_model=embed_model
        )

        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"📄 Created {len(nodes)} document chunks")
        return nodes

    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return []


def get_index():

    try:
        storage_context = get_storage_context()

        return load_index_from_storage(storage_context)
    except Exception as e:
        print(f"⚠️ Local storage not found, creating index from existing Pinecone data...")
        try:

            vector_store = get_vector_store()
            storage_context = get_storage_context()
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            return index
        except Exception as e2:
            print(f"❌ Error creating index from vector store: {e2}")
            return None

def check_index_status():

    try:
        pinecone_index = pc.Index(index_name)
        stats = pinecone_index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        
        if vector_count > 0:
            print(f"✅ Index found with {vector_count} vectors")
            return True
        else:
            print("❌ Index exists but is empty")
            return False
    except Exception as e:
        print(f"❌ Error checking index: {e}")
        return False
    


def clear_pinecone_index():
    """Delete all vectors from Pinecone index"""
    try:
        pinecone_index = pc.Index(index_name)
        

        stats = pinecone_index.describe_index_stats()
        vector_count = stats.get('total_vector_count', 0)
        print(f"🗑️ Current vectors in index: {vector_count}")
        
        if vector_count > 0:

            pinecone_index.delete(delete_all=True)
            print("✅ All vectors deleted from Pinecone index")
        else:
            print("ℹ️ Index is already empty")
            
        return True
        
    except Exception as e:
        print(f"❌ Error clearing index: {e}")
        return False

def rebuild_index():
    """Clear old data and rebuild index with new CSV processing"""
    try:
        print("🔄 Starting index rebuild process...")
        

        if not clear_pinecone_index():
            print("❌ Failed to clear index, aborting rebuild")
            return None
        

        import shutil
        if os.path.exists("./storage"):
            shutil.rmtree("./storage")
            print("🗑️ Cleared local storage")
        

        nodes = get_and_chunk_documents()
        
        if not nodes:
            print("❌ No nodes created, cannot rebuild index")
            return None
        

        storage_context = get_storage_context(for_rebuild=True)
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        

        index.storage_context.persist(persist_dir="./storage")
        
        print(f"✅ Index rebuilt successfully with {len(nodes)} nodes")
        return index
        
    except Exception as e:
        print(f"❌ Error rebuilding index: {e}")
        return None