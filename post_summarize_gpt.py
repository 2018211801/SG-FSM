#encoding:utf8
import json
from tqdm import tqdm
from state_machine2_gpt import load_json,find_json,request_gpt
import re

ROLE_INSTRUCT='You are a helpful assistant that can decompose difficult problems to subproblems and answer them correctly.'


    
def extract_paragraph(line,res):
    paras=[]   
    for para in line['paragraphs']:       
        try:
            if para['title'] in res['paragraph titles']:
                para_string=''.join(para['text']) if isinstance(para['text'],list) else para['text']
                # para_text=re.sub(r"\(.*?\)","",para_string)
                para['text']=para_string
                paras.append(para)
        except:
            pass
        
    return paras


def gen_prompts(line,paras,res):
    instruct="Documents:\nparagraphs:{}\nsubquestion and answers:{}\n Question:{}\n"
    requirement='''Answer the question reasoning step-by-step based on the Doucments. If it is a general question, please respond with 'Yes' or 'No'.Finally, you must return the title of the context, the sentence index (start from 0) of the paragraph and the concise answer no more than 10 words and explaination in the form of {"supporting_facts": [[title, sentence id], ...], "evidences": [[subject entity, relation, object entity],...], "answer":"xxx","explain":"xxxx"}. Do not reply any other words.'''
    prompt=instruct.format(paras,res,line['question']+'\n')+requirement
    return prompt

def gen_prompts2(line,paras,res):
    instruct="Paragraphs:{}\nSubquestion and probable answers:\n{}Question:{}\n"
    requirement='Answer the question by reasoning step-by-step based on the Doucments. The subquestions and answer may help and you must return the title of the paragraphs, the sentence index (start from 0) of the paragraph and the concise answer no more than 10 words and explaination in the form of {"reasoning":"xxxx","supporting_facts": [[title, sentence id], ...], "answer":"xxx"}. Do not reply any other words.'
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
            # paras.append({'title':p[0],'text':p[1]})
            try:
                paras.append({'title':p['title'],'text':p['paragraph_text']})
            except:
                paras.append({'title':p[0],'text':p[1]})
            data['paragraphs']=paras
            data['complex question']=data['question']
        # print(data.keys())
        else:
            # print(p)
            data['paragraphs']['title']="no title"
            data['paragraphs']['text']="no context"
            data['complex question']=data['question']
    return data

# def run(line,model):
#     line=parse(line)
#     res=gen_subquestion(line,model)
#     # print(f'subquestions:{res}')
#     paras=extract_paragraph(line,res['paragraph titles'])
#     prompt=gen_prompts(line,res,paras)
#     # print(prompt)
#     preds,meta=request_gpt(ROLE_INSTRUCT,prompt,500,0.8,model)
#     #print(preds[0])
#     return preds[0]

def reformat_answer(json_string,model):
    prompt='Please rewrite the illegal json text below into an legal json string in the form of {"supporting_facts": [[title, sentence id], ...], "reasoning": "xxx", "answer":"xxx"}. Text:\n'
    requirement='\nDo not reply any other words.'
    preds,meta=request_gpt('You are a helpful assistant',prompt+json_string+requirement,500,0.8,model)
    return preds[0].replace('\n','')

def post_qa(filename,origin_file,file_out,model):
    data=[]
    with open(filename,'r') as f1:
        for line in f1:
            if line !='FAILED!':
                data.append(json.loads(line))
            else:
                data.append({"answer":"FAILED!","supporting_facts":"FAILED!"})
    # data=load_json(filename)
    ori_data=load_json(origin_file)
    data=data[:1000]
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
        preds,meta=request_gpt(ROLE_INSTRUCT,prompt,500,model=model)
        # #print(preds[0])
        with open(file_out,'a',encoding='utf8') as f:
            try:
                ans=json.loads(preds[0])
                f.write(json.dumps({"answer":ans['answer'],"supporting_facts":ans['supporting_facts'],"qas":line['qas'],"paras":paras,"question":ori_line['question']},ensure_ascii=False))
                
            except:
                try:
                    try:
                        ans=json.loads(reformat_answer(preds[0],model))
                        f.write(json.dumps({"answer":ans['answer'],"supporting_facts":ans['supporting_facts'],"qas":line['qas'],"paras":paras,"question":ori_line['question']},ensure_ascii=False))
                
                    except:
                        preds,meta=request_gpt(ROLE_INSTRUCT,prompt,500,model=model)
                        ans=json.loads(preds[0])
                        f.write(json.dumps({"answer":ans['answer'],"supporting_facts":ans['supporting_facts'],"qas":line['qas'],"paras":paras,"question":ori_line['question']},ensure_ascii=False))
                
                except:
                    f.write("FAILED!")
                
                # f.write(json.dumps({"answer":ans['answer'],"supporting_facts":ans['supporting_facts'],"qas":line['qas'],"paras":paras,"question":ori_line['question']},ensure_ascii=False))
                
                    
                    # ans=json.loads(reformat_answer(preds[0],model))
                    # f.write(json.dumps({"answer":ans['answer'],"supporting_facts":ans['supporting_facts'],"qas":line['qas'],"paras":paras,"question":ori_line['question']},ensure_ascii=False))
                f.write('\n')
        f.close()
    print('finish')

if __name__=='__main__':
    # model='gpt-4-1106-preview'
    model='gpt-3.5-turbo-1106'
    m_name='gpt35_tri'
    hot='/cognitive_comp/NAME/projects/MHQA/900/hotpot/hotpot_clean1000.jsonl'   
    wiki='/cognitive_comp/NAME/projects/MHQA/900/2wiki/wiki_clean1000.jsonl'
    hot_o=f'/cognitive_comp/NAME/projects/MHQA/results/automat/hotpot/{m_name}.jsonl6' 
    wiki_o=f'/cognitive_comp/NAME/projects/MHQA/results/automat/wiki/{m_name}.jsonl6'
    wiki_p=f'/cognitive_comp/NAME/projects/MHQA/results/automat/wiki/{m_name}_post.jsonl6'
    hot_p=f'/cognitive_comp/NAME/projects/MHQA/results/automat/hotpot/{m_name}_post.jsonl6'
    musi='/cognitive_comp/NAME/projects/MHQA/900/musi/musi_clean1000.jsonl'
    musi_o=f'/cognitive_comp/NAME/projects/MHQA/results/automat/musi/{m_name}.jsonl6'
    musi_p=f'/cognitive_comp/NAME/projects/MHQA/results/automat/musi/{m_name}_post.jsonl6'
    # for i1,i2,i3 in zip([hot_o,wiki_o,musi_o],[hot,wiki,musi],[hot_p,wiki_p,musi_p]):
    # for i1,i2,i3 in zip([hot_o],[hot],[hot_p]): 
    for i1,i2,i3 in zip([wiki_o],[wiki],[wiki_p]): 
        print(i3)
        post_qa(i1,i2,i3,model)
        print(i3)
        