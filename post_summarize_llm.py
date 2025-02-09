#encoding:utf8
import json
from tqdm import tqdm
from state_machine_qwen import load_json,find_json
import re
from vllm import LLM, SamplingParams
from vllm_deepseek67 import request_llm,init_llm
ROLE_INSTRUCT='You are a helpful assistant that can decompose difficult problems to subproblems and answer them correctly.'
    
def extract_paragraph(line,res):
    paras=[]
    # patten=re.compile()
    # print(line)
    for para in line['paragraphs']:
        
        
        if isinstance(para, dict):    
            if para['title'] in res['paragraph titles']:
                para_string=''.join(para['text']) if isinstance(para['text'],list) else para['text']
                # para_text=re.sub(r"\(.*?\)","",para_string)
                para['text']=para_string
                paras.append(para)
        else:
            para={}
            para['text']="no text"
            paras.append(para)
    return paras


def gen_subquestion(line,model):
    instruct="This is a two-hop to four-hop reasoning question-answering task that requires decomposing the questions into simple, answerable single-hop questions. You are going to answer a question and find out the title of supporting paragraphs precisely, given several paragraphs that may contain the correct answer. \n Question:{}\n Paragraphs:{}\n"
    # requirement='Please decompose the complex question and generate the simple question and its answer iteratively in the form of [{"subquestion":xxx,"answer":xxx,"paragraph title":xxx},{"subquestion":xxx,"answer":xxx,"paragraph title"xxx}]'
    requirement='Please decompose the complex question and generate the simple question and its answer iteratively in the form of [{"subquestion": Q1,"paragraph title": xxx,"answer": A1},{"subquestion":Q2,"paragraph title":xxx,"answer":A2}]'
    prompt=instruct.format(line['question'],line['paragraphs'])+requirement
    preds,meta=request_llm(ROLE_INSTRUCT,[prompt],500,model)
    #print(preds[0])
    return json.loads(preds[0])

def gen_prompts(line,paras,res):
    instruct="Documents:\nparagraphs:{}\nsubquestion and answers:{}\n Question:{}\n"
    requirement='''Answer the question reasoning step-by-step based on the Doucments. If it is a general question, please respond with 'Yes' or 'No'.Finally, you must return the title of the context, the sentence index (start from 0) of the paragraph and the concise answer no more than 10 words and explaination in the form of {"supporting_facts": [[title, sentence id], ...], "evidences": [[subject entity, relation, object entity],...], "answer":"xxx","explain":"xxxx"}. Do not reply any other words.'''
    prompt=instruct.format(paras,res,line['question']+'\n')+requirement
    return prompt

def gen_prompts2(line,paras,res):
    instruct="Paragraphs:{}\nSubquestion and probable answers:\n{}Question:{}\n"
    requirement='Answer the question by reasoning step-by-step based on the Doucments. The subquestions and answer may help and you must return the title of the paragraphs, the sentence index (start from 0) of the paragraph and the concise answer no more than 10 words and explaination in the form of {"reasoning":"xxxx","supporting_facts": [[title, sentence id], ...], "answer":"Please respond with single words only"}. Do not reply any other words.'
    qas=""
    for r in res:
        if "question" in r:
            qas+=f'<{r["question"]} {r["answer"]}>\n'
    # print("qas:",qas)
    prompt=instruct.format(paras,qas,line['question']+'\n')+requirement
    return prompt
    

def parse(data):
    paras=[]
    # print(data)
    for p in data['context']:
        # print(len(p))
        if len(p)!=0:
            try:
                paras.append({'title':p['title'],'text':p['paragraph_text']})
            except:
                paras.append({'title':p[0],'text':p[1]})
            # paras.append({'title':p[0],'text':p[1]})
            # paras.append({'title':p['title'],'text':p['paragraph_text']})
            data['paragraphs']=paras
            data['complex question']=data['question']
        # print(data.keys())
        else:
            # print(p)
            data['paragraphs']['title']="no title"
            data['paragraphs']['text']="no context"
            data['complex question']=data['question']
    return data

def run(line,model):
    line=parse(line)
    res=gen_subquestion(line,model)
    # print(f'subquestions:{res}')
    paras=extract_paragraph(line,res['paragraph titles'])
    prompt=gen_prompts(line,res,paras)
    # print(prompt)
    preds,meta=request_llm(ROLE_INSTRUCT,[prompt],500,model)
    #print(preds[0])
    return preds[0]

def reformat_answer(json_string,model):
    prompt='Please rewrite the illegal json text below into an legal json string in the form of {"supporting_facts": [[title, sentence id], ...], "reasoning": "xxx", "answer":"xxx"}. Text:\n'
    requirement='\nDo not reply any other words.'
    # preds,meta=request_llm(ROLE_INSTRUCT,[prompt],500,model,history=history[-4:])
    preds,meta=request_llm('You are a helpful assistant',[prompt+json_string+requirement],500,model)
    return preds[0].replace('\n','')

def post_qa(filename,origin_file,file_out,model):
    data=load_json(filename)
    ori_data=load_json(origin_file)
    # data=data[:900]
    j=0
    try:
        fout = open(file_out, mode="r")            
        for line in fout:
            if fout!=None:
                j+=1
        #147          
        fout.close()
        # print(f'已推理到{j}case')  
    except FileNotFoundError:
        j=0    
    
    for i,line in enumerate(tqdm(data)):
        if i<j:
            continue
        if line == 'FAILED!':
            with open(file_out,'a',encoding='utf8') as f:
                f.write("FAILED!")
                f.write('\n')
                f.close()
            continue
                        # print(f'case {i+1}')
        # if i<=84:
        #     continue
        ori_line=parse(ori_data[i])
        if len(ori_line['context']) == 0:
            ori_line['paragraphs']=['no title','no text']
            # ori_line['context'][0]='no title'
            # ori_line['context'][1]='no text'
        # print(ori_line)    
        paras=extract_paragraph(ori_line,line)
        prompt=gen_prompts2(ori_line,paras,line['qas'])
        # print(prompt)
        preds,meta=request_llm(ROLE_INSTRUCT,[prompt],500,model)
        # preds,meta=request_gpt(ROLE_INSTRUCT,prompt,500,0.8,model)
        # #print(preds[0])
        try:
            ans=json.loads(preds[0])
        except:
            print(reformat_answer(preds[0],model))
            try:
                preds=reformat_answer(preds[0],model)
                ans=json.loads(preds[0])
            except:
                index1=preds[0].find('supporting_facts')
                index2=preds[0].find('reasoning')
                index3=preds[0].find('answer')
                if index3 !=-1 and index1 !=-1 and index2 !=-1:
                    ans={}
                    ans['supporting_facts']=preds[0][index1+len('supporting_facts'):index2]
                    ans['reasoning']=preds[0][index2+len('reasoning'):index3]
                    ans['answer']=preds[0][index3+len('reasoning'):]

        with open(file_out,'a',encoding='utf8') as f:
            try:
                f.write(json.dumps({"answer":ans['answer'],"supporting_facts":ans['supporting_facts'],"reasoning":ans['reasoning'],"qas":line['qas'],"paras":paras,"question":ori_line['question']},ensure_ascii=False))
            except:
                try:
                    ans=json.loads(reformat_answer(preds[0],model))
                    f.write(json.dumps({"answer":ans['answer'],"supporting_facts":ans['supporting_facts'],"reasoning":ans['reasoning'],"qas":line['qas'],"paras":paras,"question":ori_line['question']},ensure_ascii=False))
                except:
                    f.write("FAILED!")
            f.write('\n')
            f.close()
    print('finish')

if __name__=='__main__':
    hot='/cognitive_comp/NAME/projects/MHQA/900/hotpot/hotpot_clean1000.jsonl'
    musi='/cognitive_comp/NAME/projects/MHQA/900/musi/musi_clean1000.jsonl'
    wiki='/cognitive_comp/NAME/projects/MHQA/900/2wiki/wiki_clean1000.jsonl'
    hot_o='/cognitive_comp/NAME/projects/MHQA/results/automat/hotpot/deepseek67.jsonl6'
    musi_o='/cognitive_comp/NAME/projects/MHQA/results/automat/musi/deepseek67.jsonl6'
    wiki_o='/cognitive_comp/NAME/projects/MHQA/results/automat/wiki/deepseek67.jsonl6'
    hot_p='/cognitive_comp/NAME/projects/MHQA/results/automat/hotpot/deepseek67_post.jsonl6'
    musi_p='/cognitive_comp/NAME/projects/MHQA/results/automat/musi/deepseek67_post.jsonl6'
    wiki_p='/cognitive_comp/NAME/projects/MHQA/results/automat/wiki/deepseek67_post.jsonl6'
    # for i,o in zip([hot],[hot_o]):
    # model_name = "/cognitive_comp/NAME/dmodels/Qwen-72B-Chat"
    # model_name="/home/NAME/.cache/modelscope/hub/Qwen/Qwen-7B-Chat"
    # model_name="/cognitive_comp/NAME/dmodels/Qwen-72B-Chat"
    model_name = "/cognitive_comp/NAME/dmodels/deepseek-llm-67b-chat"
    # model_name="/cognitive_comp/NAME/dmodels/Llama-2-70b-chat-ms"
    # model_name = "/home/NAME/.cache/modelscope/hub/Qwen/Qwen-14B-Chat"
    llm=init_llm(model_name,tp_size=4)
    # post_qa(hot_o,musi_o,wiki_o,wiki,wiki_p)
    for i1,i2,i3 in zip([hot_o,musi_o,wiki_o],[hot,musi,wiki],[hot_p,musi_p,wiki_p]):
        print(i3)
        post_qa(i1,i2,i3,llm)
        print(i3)
      