import os
import sys
import json
import time
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
# from vllm_wrapper import vLLMWrapper
sys.path.append('./')
from conversation import get_conv_template
   
def get_prompt(query,new_history):
    # print("query:::",query)
    
    if new_history is not None:
        # print('进入history')  
        conv = get_conv_template("qwen-7b-chat")
        for his in new_history:
            if his['role'] == 'user':      
                conv.append_message(conv.roles[0], his['content'])
            else:
                conv.append_message(conv.roles[1], his['content'])
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() 
    else:
        conv = get_conv_template("qwen-7b-chat")
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    return prompt

def init_llm(model,tp_size=2):  

    llm = LLM(model=model, trust_remote_code=True, gpu_memory_utilization=0.8, tensor_parallel_size=tp_size)
    return llm

def request_llm(ROLE_INSTRUCT,prompt,max_tokens,llm,history=[]):
    # fout=open('qwen72prompt_pred.log','a')
    prompt_p=[get_prompt(query,history) for query in prompt]
    sampling_params = SamplingParams(
        stop=["<|im_end|>", "<|im_start|>",],
        top_k= -1, 
        repetition_penalty=1.1,
        temperature=0.8, 
        top_p=0.8,
        # stop=["<|endodtext|>", "<|im_end|>","<|im_sep|>"] ,
        max_tokens=max_tokens)
    # print(prompt_p)    
    output_l=llm.generate(prompt_p, sampling_params)    
    preds = output_l[0].outputs[0].text    
    # fout.write(prompt_p[0])
    # fout.write(preds[0])
    # fout.write('\n')
    # fout.close()
    if preds is None:
        output_l=llm.generate(prompt_p, sampling_params)    
        preds = output_l[0].outputs[0].text   
    # print(f'preds:::::{preds[0]}')
    return [preds],[]

def request_llm_copy(ROLE_INSTRUCT,prompt,max_tokens,model='',history=[]):
    
    # tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    sampling_params = SamplingParams(
        stop=["<|im_end|>", "<|im_start|>",],
        top_k= -1, 
        repetition_penalty=1.1,
        temperature=0.8, 
        top_p=0.8,
        # stop=["<|endodtext|>", "<|im_end|>","<|im_sep|>"] ,
        max_tokens=max_tokens)
    tp_size=4
    llm = LLM(model=model, trust_remote_code=True, gpu_memory_utilization=0.8, tensor_parallel_size=tp_size)
    
    # prompts_1 = [tokenizer.apply_chat_template([{"role": "system", "content":"you are a helpful ai assistant developed by openai. You are skilled in planning, reasoning, decomposing complex problems and answering questions based on reference paragraph."},{"role": "user", "content": {messages}}], add_generation_prompt=True, tokenize=False  ) for messages in pprompts]           
    prompt_p=[get_prompt(query,history) for query in prompt]
    print(f'进入qwen{prompt_p}')
    
    output_l=llm.generate(prompt_p, sampling_params)    
    preds = output_l[0].outputs[0].text
    print(f'preds[0]::{[preds][0]}')
    print(f'preds:{preds}')
    return [preds],[]            
prompt_6r_pro_en='''This is a two-hop to four-hop reasoning question-answering task that requires decomposing the questions into simple, answerable single-hop questions. The decomposition process involves four types of questions: comparison, inference, compositional, and bridge-comparison. There are six specific decomposition steps in total, denoted by Q* representing the decomposed subproblems. The steps are as follows:

First, Q1 -> Q2
Second, Q1 -> Q2 -> Q3
Third, Q1 -> Q2 -> Q3
Fourth, (Q1&Q2) -> Q3
Fifth, (Q1&Q2) -> Q3; Q3 -> Q4
Sixth, Q1 -> Q2; (Q2&Q3) -> Q4
The process involves first determining the type of question and then identifying the decomposition process type. It's important to note that the decomposition of questions cannot be provided all at once; it must be done step by step. Each subproblem needs to be decomposed and answered before moving on to the next one, as there is interdependence between the subproblems.Finally, you must return the title of the context, the sentence index (start from 0) of the paragraph and the concise answer and explaination in the form of 
{"explain":"xxxx","supporting_facts": [[title, sentence id], ...], "evidences": [[subject entity, relation, object entity],...],"answer":"no sentence and no more than 10 words "}. 
Do not reply any other words.'''
sys_prompt_nodecom_step_notriple='''Answer the question according to the context,Let's think step by step, and explain your reasoning process. You must return in the form of {"explain":"xxxx","answer":answer}. Do not reply any other words.'''
prompt_nodecom_normal_notri='''Answer the question according to the context. You must return in the form of {"explain":"xxxx","answer":answer}. Do not reply any other words.'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT answer generation.")
    parser.add_argument("-q", "--question", default='/cognitive_comp/NAME/projects/MHQA/1000/hotpot/hotpot_clean1000.jsonl')
    parser.add_argument("-o", "--output", default='/cognitive_comp/NAME/projects/MHQA/1000/2wiki/wiki_clean1000_6r_g35.jsonl', type=str)    
    args = parser.parse_args()  
    prompts = []
    hot='/cognitive_comp/NAME/projects/MHQA/1000/hotpot/hotpot_clean1000_shuffle.jsonl'
    musi='/cognitive_comp/NAME/projects/MHQA/1000/musi/musi_clean1000_shuffle.jsonl'
    wiki='/cognitive_comp/NAME/projects/MHQA/1000/2wiki/wiki_clean1000_shuffle.jsonl'
    hot_o='/cognitive_comp/NAME/projects/MHQA/results/qwen72/hotpot/'
    musi_o='/cognitive_comp/NAME/projects/MHQA/results/qwen72/musi/'
    wiki_o='/cognitive_comp/NAME/projects/MHQA/results/qwen72/wiki/'
    # model_name = "/cognitive_comp/NAME/dmodels/Qwen-72B-Chat"
    model_name = "/home/NAME/.cache/modelscope/hub/Qwen/Qwen-14B-Chat"
    # model_name="/home/NAME/.cache/modelscope/hub/Qwen/Qwen-7B-Chat"
    preds,meta=request_llm('ROLE_INSTRUCT',['who are you?'],200,model=model_name)
    print(preds[0])
    