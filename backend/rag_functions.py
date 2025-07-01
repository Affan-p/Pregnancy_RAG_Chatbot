import os
import requests
from backend.utils import get_and_chunk_documents, llm, embed_model, get_index
from backend.utils import Settings 
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.settings import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import QueryFusionRetriever
import json


Settings.llm = llm
Settings.embed_model = embed_model


index = get_index()
hybrid_retriever = None
vector_retriever = None
bm25_retriever = None

if index:
    try:
        
        vector_retriever = index.as_retriever(similarity_top_k=15)
        print("✅ Vector retriever initialized successfully")
        
        
        all_nodes = index.docstore.docs
        if len(all_nodes) == 0:
            print("⚠️ Warning: No documents found in index, skipping BM25 retriever")
            hybrid_retriever = vector_retriever  
        else:
            
            has_text_content = False
            for node_id, node in all_nodes.items():
                if hasattr(node, 'text') and node.text and node.text.strip():
                    has_text_content = True
                    break
            
            if not has_text_content:
                print("⚠️ Warning: No text content found in documents, skipping BM25 retriever")
                hybrid_retriever = vector_retriever  
            else:
                try:
                    
                    print("🔄 Creating BM25 retriever...")
                    bm25_retriever = BM25Retriever.from_defaults(
                        docstore=index.docstore, 
                        similarity_top_k=15,
                        verbose=False
                    )
                    print("✅ BM25 retriever initialized successfully")
                    
                    
                    hybrid_retriever = QueryFusionRetriever(
                        retrievers=[vector_retriever, bm25_retriever],
                        similarity_top_k=20,
                        num_queries=1,  
                        mode="reciprocal_rerank",  
                        use_async=False,
                    )
                    print("✅ Hybrid retriever initialized successfully")
                    
                except Exception as e:
                    print(f"❌ Warning: Could not initialize BM25 retriever: {e}")
                    print("🔄 Falling back to vector-only retrieval")
                    hybrid_retriever = vector_retriever
        
    except Exception as e:
        print(f"❌ Warning: Could not initialize retrievers: {e}")
        hybrid_retriever = None
        vector_retriever = None
        bm25_retriever = None
else:
    print("❌ Warning: Could not initialize retrievers - index is None")

def call_groq_api(prompt):
    """Call Groq API instead of LM Studio"""
    try:
        
        response = Settings.llm.complete(prompt)
        return str(response)
    except Exception as e:
        print(f"❌ Groq API call failed: {e}")
        raise e

def get_direct_answer(question, symptom_summary, conversation_context="", max_context_nodes=8, is_risk_assessment=True):
    """Get answer using hybrid retriever with retrieved context"""
    
    print(f"🎯 Processing question: {question}")
    
    if not hybrid_retriever:
        return "Error: Retriever not available. Please check if documents are properly loaded in the index."
    
    try:
        
        print("🔍 Retrieving with available retrieval method...")
        retrieved_nodes = hybrid_retriever.retrieve(question)
        print(f"📊 Retrieved {len(retrieved_nodes)} nodes")
        
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return f"Error during document retrieval: {e}. Please check your document index."
    
    if not retrieved_nodes:
        return "No relevant documents found for this question. Please ensure your medical knowledge base is properly loaded and consult your healthcare provider for medical advice."
    
    
    try:
        reranker = SentenceTransformerRerank(
            model='cross-encoder/ms-marco-MiniLM-L-2-v2',
            top_n=max_context_nodes,
        )
        
        reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_str=question)
        print(f"🎯 After reranking: {len(reranked_nodes)} nodes")
        
    except Exception as e:
        print(f"❌ Reranking failed: {e}, using original nodes")
        reranked_nodes = retrieved_nodes[:max_context_nodes]
    
    
    filtered_nodes = []
    pregnancy_keywords = ['pregnancy', 'preeclampsia', 'gestational', 'trimester', 'fetal', 'bleeding', 'contractions', 'prenatal']
    
    for node in reranked_nodes:
        node_text = node.get_text().lower()
        if any(keyword in node_text for keyword in pregnancy_keywords):
            filtered_nodes.append(node)
    
    if filtered_nodes:
        reranked_nodes = filtered_nodes[:max_context_nodes]
        print(f"🔍 After pregnancy keyword filtering: {len(reranked_nodes)} nodes")
    else:
        print("⚠️ No pregnancy-related content found, using original nodes")
    
    
    context_chunks = []
    total_chars = 0
    max_context_chars = 6000  
    
    for node in reranked_nodes:
        node_text = node.get_text()
        if total_chars + len(node_text) <= max_context_chars:
            context_chunks.append(node_text)
            total_chars += len(node_text)
        else:
            remaining_chars = max_context_chars - total_chars
            if remaining_chars > 100:
                context_chunks.append(node_text[:remaining_chars] + "...")
            break
    
    context_text = "\n\n---\n\n".join(context_chunks)
    
    
    if is_risk_assessment:
        prompt = f"""You are the GraviLog Pregnancy Risk Assessment Agent. Use ONLY the context below—do not invent or add any new medical facts.

    SYMPTOM RESPONSES:
    {symptom_summary}

    MEDICAL KNOWLEDGE:
    {context_text}

    Respond ONLY in this exact format (no extra text):

    🏥 Risk Assessment Complete  
    **Risk Level:** <Low/Medium/High>  
    **Recommended Action:** <from KB's Risk Output Labels>  

    🔬 Rationale:  
    <One or two sentences citing which bullet(s) from the KB triggered your risk level.>"""

    else:
        
        prompt = f"""You are a pregnancy health assistant. Based on the medical knowledge below, answer the user's question about pregnancy symptoms and conditions.

    USER QUESTION: {question}

    CONVERSATION CONTEXT:
    {conversation_context}

    CURRENT SYMPTOMS REPORTED:
    {symptom_summary}

    MEDICAL KNOWLEDGE:
    {context_text}

    Provide a clear, informative answer based on the medical knowledge. Always mention if symptoms require medical attention and provide risk level (Low/Medium/High) when relevant."""
    
    try:
        print("🤖 Generating response with Groq API...")
        response_text = call_groq_api(prompt)
        return response_text
        
    except Exception as e:
        print(f"❌ LLM response failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating response: {e}"

def get_answer_with_query_engine(question):
    """Alternative approach using LlamaIndex query engine with hybrid retrieval"""
    try:
        print(f"🎯 Processing question with query engine: {question}")
        
        if index is None:
            return "Error: Could not load index"
        
        
        if hybrid_retriever:
            query_engine = RetrieverQueryEngine.from_args(
                retriever=hybrid_retriever,
                response_synthesizer=get_response_synthesizer(
                    response_mode="compact",
                    use_async=False
                ),
                node_postprocessors=[
                    SentenceTransformerRerank(
                        model='cross-encoder/ms-marco-MiniLM-L-2-v2',
                        top_n=5
                    )
                ]
            )
        else:
            
            query_engine = index.as_query_engine(
                similarity_top_k=10,
                response_mode="compact"
            )
        
        print("🤖 Querying with engine...")
        response = query_engine.query(question)
        
        return str(response)
        
    except Exception as e:
        print(f"❌ Query engine failed: {e}")
        import traceback
        traceback.print_exc()
        return f"Error with query engine: {e}. Please check your setup and try again."