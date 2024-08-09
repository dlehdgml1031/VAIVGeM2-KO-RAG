# -*- coding: utf-8 -*-
"""
Created on 2024.07.13
Description: This is the main file for VAIVGeM2 KO RAG project.
Author: DONGHEE LEE
"""

# Importing libraries
from utils.text_processing import extract_question_answer

import os
import sys
from typing import List, Dict, Any, Tuple

import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM

from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter




# from langchain import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.llms import HuggingFacePipeline


def inference_llm(
    model_id: str,
    input_querys: List,
    gen_token_len:int = 256,
    chat_template_flag: bool = True,
    quantization_flag: bool = False,
    flash_attention_flag: bool = False,
):  
    
    # Define the device if cuda is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Quantization config
    if quantization_flag:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_compute_dtype = torch.float16
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load LLM model
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = model_id,
        torch_dtype = torch.float16,
        quantization_config = quantization_config if quantization_flag else None,
        # low_cpu_mem_usage = False, # use as much memory as we can
        # attn_implementation = attn_implementation, # use flash attention
    )
    
    if quantization_flag:
        pass
    else:
        llm_model.to(device)
    
    for input_query in input_querys:
            # Use dialogue template if chat_template_flag is True; otherwise, use input query directly
        if chat_template_flag:
            messages = [
                {
                    "role" : "user",
                    "content" : input_query
                }
            ]
            
            prompt = tokenizer.apply_chat_template(
                conversation = messages,
                tokenize = False,
                add_generation_prompt = True,
            )
            
            input_ids = tokenizer(prompt, return_tensors = "pt").to(device)
        else:
            input_ids = tokenizer(input_query, return_tensors="pt").to(device)
        
        # Remove token_type_ids
        input_ids.pop("token_type_ids", None)
        
        # Generate answer
        outputs = llm_model.generate(
            **input_ids,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
            max_new_tokens = gen_token_len, # maximum number of tokens to generate
        )
        
        outputs_decoded = tokenizer.decode(
            outputs[0],
            skip_special_tokens = True,
        )
        
        # Extract question and answer from the outputs_decoded
        q, a = extract_question_answer(outputs_decoded)
        print(f'Q: {q}')
        print(f'A: {a}')


def pdf2Vec(
    document_path: str,
    embedding_model_id: str,
):
    pdf_loader = PyPDFLoader(document_path)
    
    pages = pdf_loader.load_and_split()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
    )
    
    docs = text_splitter.split_documents(pages)
    
    # encode_kwargs = {'normalize_embeddings': True}
    embedding_model = HuggingFaceEmbeddings(
        model_name = embedding_model_id,
        # encode_kwargs = encode_kwargs
    )
    
    db = FAISS.from_documents(docs, embedding_model)
    
    return db


def define_retriever(
    db: FAISS,
    search_type: str = 'similarity',
    search_kwargs: Dict = {'k': 5},
):
    return db.as_retriever(search_type = search_type, search_kwargs = search_kwargs)


def generate_prompt_template():
    prompt_template = """
    ### [INST]
    Instruction: Answer the question based on your knowledge.
    Here is context to help:
    
    {context}

    ### QUESTION:
    {question}

    [/INST]
    """
    
    prompt = PromptTemplate(
        input_variables = ['context', 'question'],
        template = prompt_template,
    )
    
    return prompt

def run_rag(
    generation_model_id:str,
    embedding_model_id:str,
    document_path:str,
    input_query:str,
    gen_token_len:int = 256,
):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(generation_model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = generation_model_id,
        torch_dtype = torch.float16,
        device_map = 'auto'
    )
    
    # llm_model.to(device)
    
    text_generation_pipeline = pipeline(
        model = llm_model,
        tokenizer = tokenizer,
        task = 'text-generation',
        return_full_text = True,
        max_new_tokens = gen_token_len,
    )
    
    llm_pipeline = HuggingFacePipeline(pipeline = text_generation_pipeline)
    prompt = generate_prompt_template()
    
    llm_chain = LLMChain(
        llm = llm_pipeline,
        prompt = prompt,
    )
    
    faiss_db = pdf2Vec(
        document_path = document_path,
        embedding_model_id = embedding_model_id,
    )
    
    retriever = define_retriever(db = faiss_db)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
            | llm_chain
    )
    
    result = rag_chain.invoke(input_query)
    
    print(f'Q: {input_query}')
    
    for i in result['context']:
        print(f"주어진 근거: {i.page_content} / 출처: {i.metadata['source']} - {i.metadata['page']} \n\n")

    print(f"A: {result['text']}")



if __name__ == "__main__":
    # Set the environment variable for CUDA
    # export CUDA_VISIBLE_DEVICES=5
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    model_dir = "/mnt/nvme01/huggingface/models"
    
    run_rag(
        generation_model_id = model_dir + "/Vaiv/GeM2-Llamion-14B-Chat",
        embedding_model_id = '/home/dlehdgml1031/code/VAIVGeM2-KO-RAG/model/ko-sbert-nli',
        document_path = 'document/finance/통화신용정책보고서(2024년 3월).pdf',
        input_query = "2023년 4/4분기 중에 한국은행이 유동성조절을 위해 취한 대응책들에는 어떤 것들이 있으며, 그 결과로 이루어진 지준공급의 변화는 어떤 모습이었나요?",
    )

    # inference_llm(
    #     model_id = model_dir + "/Vaiv/GeM2-Llamion-14B-Chat",
    #     input_querys = ['재난 상황에서는 어떻게 해야되는지 설명해줘.'],
    # )