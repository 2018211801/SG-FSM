
# encoding:utf8
import sys
import os
sys.path.append('./')
sys.path.append('/cognitive_comp/NAME/projects')
import time
import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm_llama70 import request_llm,init_llm
from state_machine_prompt import main
ROLE_INSTRUCT='You are a helpful assistant that can decompose difficult problems to subproblems and find the final answe correctly.'


if __name__=='__main__':

    hot='/cognitive_comp/NAME/projects/MHQA/900/hotpot/hotpot_clean1000.jsonl'
    musi='/cognitive_comp/NAME/projects/MHQA/900/musi/musi_clean1000.jsonl'
    wiki='/cognitive_comp/NAME/projects/MHQA/900/2wiki/wiki_clean1000.jsonl'
    hot_o='/cognitive_comp/NAME/projects/MHQA/results/automat/hotpot/llama70.jsonl6'
    musi_o='/cognitive_comp/NAME/projects/MHQA/results/automat/musi/llama70.jsonl6'
    wiki_o='/cognitive_comp/NAME/projects/MHQA/results/automat/wiki/llama70.jsonl6'
   
    model_name="/cognitive_comp/NAME/dmodels/Llama-2-70b-chat-ms"

    llm=init_llm(model_name,tp_size=4)
    # main(wiki,wiki_o,4,llm)
    # main(hot,hot_o,2,llm)
    for i,o in zip([hot,wiki,musi],[hot_o,wiki_o,musi_o]):
        main(i,o,4,llm)
 
