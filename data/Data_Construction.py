import pandas as pd
import numpy as np
import xmltodict
import os 
import re
import traceback

# utterance concat
def uttr_concat(df):
    ids = df['id'].unique()
    
    all_list = []
    
    for i in ids:
        # get all rows whose  id == i
        temp_df = df[df['id'] == i]
        temp_df.reset_index(inplace=True)
        
        # record the other columns
        other_columns = list(temp_df)[3::]
        f_df = temp_df.loc[0][other_columns].tolist()

        # the utterance list
        text_list = temp_df['utterance'].tolist()

        temp_text = ''

        result_list = []
        
        def uttr_len(uttr):
            uttr_list = uttr.split(' ')
            return len(uttr_list)

        for j,text in enumerate(text_list):
            t = temp_text + text
            if uttr_len(t) > 512:
                # append temp_text into result_list
                result_list.append([i, temp_text] + f_df)
                # let the new text become the new temp_text
                temp_text = text
            else:
                # the last uttr,append to result_list
                if j == (len(text_list) - 1):
                    result_list.append([i, t] + f_df)
                # let t = new temp_text
                temp_text = t

        all_list+=result_list
    
    all_df = pd.DataFrame(all_list, columns=list(df))
    return all_df

# process myPersonality dataset
def myPersonality_process():
    # read data
    myPersonality = pd.read_csv('Source_Data/mypersonality_final.csv',sep = ',')
    # rename columns
    myPersonality.rename(columns={'#AUTHID':'id','STATUS':'utterance'},inplace=True)
    # concat rows
    myPersonality = uttr_concat(myPersonality)
    # change the class type
    myPersonality['cAGR'] = myPersonality['cAGR'].apply(lambda x: 1 if x == 'y' else 0)
    myPersonality['cCON'] = myPersonality['cCON'].apply(lambda x: 1 if x == 'y' else 0)
    myPersonality['cEXT'] = myPersonality['cEXT'].apply(lambda x: 1 if x == 'y' else 0)
    myPersonality['cOPN'] = myPersonality['cOPN'].apply(lambda x: 1 if x == 'y' else 0)
    myPersonality['cNEU'] = myPersonality['cNEU'].apply(lambda x: 1 if x == 'y' else 0)
    myPersonality['utterance'] = myPersonality['utterance'].apply(remove_link)
    return myPersonality

# process pan dataset
def pan_process():
    Pan_train_path = 'Source_Data/pan15-author-profiling-training-dataset-2015-04-23/pan15-author-profiling-training-dataset-english-2015-04-23/'
    Pan_test_path = 'Source_Data/pan15-author-profiling-test-dataset-2015-04-23/pan15-author-profiling-test-dataset2-english-2015-04-23/'
    Pan_train_list = []
    Pan_test_list = []
    # open the txt file
    with open(Pan_train_path + 'truth.txt', 'r') as f:
        for line in f.readlines():
            tmp_train_list = line.strip().split(':::')
            Pan_train_list.append(tmp_train_list)
    with open(Pan_test_path + 'truth.txt', 'r') as f:
        for line in f.readlines():
            tmp_test_list = line.strip().split(':::')
            Pan_test_list.append(tmp_test_list)
    df_Pan_train = pd.DataFrame(Pan_train_list, columns = ['u_id', 'gender', 'age', 'E', 'N', 'A', 'C', 'O'])
    df_Pan_test = pd.DataFrame(Pan_test_list, columns = ['u_id', 'gender', 'age', 'E', 'N', 'A', 'C', 'O'])
    
    # get utterance
    def get_train_uttr_by_id(uid):
        uttr_xml = open(Pan_train_path + uid + '.xml', 'r')
        xmlDict = xmltodict.parse(uttr_xml.read())
        uttr = '|'.join(xmlDict['author']['document'])
        uttr_xml.close()
        return uttr
    
    def get_test_uttr_by_id(uid):
        uttr_xml = open(Pan_test_path + uid + '.xml', 'r')
        xmlDict = xmltodict.parse(uttr_xml.read())
        xml_df = pd.DataFrame(xmlDict)
        uttr = '|'.join(xmlDict['author']['document'])
        uttr_xml.close()
        return uttr           
    df_Pan_train['utterance'] = df_Pan_train['u_id'].apply(get_train_uttr_by_id)
    df_Pan_test['utterance'] = df_Pan_test['u_id'].apply(get_test_uttr_by_id)
    
    # concat train and test
    df_whole = pd.concat([df_Pan_train, df_Pan_test])
    
    # rename column
    df_whole.rename(columns={'u_id':'id'},inplace=True)
    
    # split original uttr
    df_uttr = df_whole['utterance'].str.split('|',expand = True)
    df_uttr = df_uttr.stack()
    df_uttr = df_uttr.reset_index(level=1,drop=True)
    df_uttr.name='utterance'
    df_whole = df_whole.drop(['utterance'], axis=1).join(df_uttr)
    
    # change the pos of utterance
    utterance = df_whole['utterance']
    df_whole = df_whole.drop('utterance',axis = 1)
    df_whole.insert(1,'utterance',utterance)
    
    # concat rows
    df_whole = uttr_concat(df_whole)
    
    # get trait label
    def get_trait_label(score):
        cls_threshold = 0.2
        if eval(score) >= cls_threshold:
            return 1
        else: return 0
    # cAGR,cCON,cEXT,cOPN,cNEU
    df_whole['cAGR'] = df_whole['A'].apply(get_trait_label)
    df_whole['cCON'] = df_whole['C'].apply(get_trait_label)
    df_whole['cEXT'] = df_whole['E'].apply(get_trait_label)
    df_whole['cOPN'] = df_whole['O'].apply(get_trait_label)
    df_whole['cNEU'] = df_whole['N'].apply(get_trait_label)
    
    def pan_data_clean(uttr):   
        uttr = re.sub('@username','',uttr)
        uttr = re.sub('#+','',uttr)
        uttr = re.sub('\n', ' ',uttr)
        uttr = re.sub('w/ [0-9]','',uttr)
        uttr = re.sub('w/ ','',uttr)
        uttr = re.sub('at (.)*? \[pic\]:', '', uttr)
        uttr = re.sub('4 RT, (.)*? Symposium:','',uttr)
        uttr = re.sub('\[pic\]:', '',uttr)
        uttr_split = uttr.split(' ')
        uttr = ' '.join(uttr_split)
        return uttr
    df_whole['utterance'] = df_whole['utterance'].apply(pan_data_clean)
    df_whole = df_whole.drop_duplicates(['utterance'])
    df_whole = df_whole.reset_index(drop=True)
    df_whole['utterance'] = df_whole['utterance'].apply(remove_link)
    return df_whole


# process friends dataset
def friends_process():
    df = pd.read_csv('Source_Data/friends-personality.csv')
    df['raw_text'] = df['raw_text'].apply(lambda x: [[i.split('</b>:')[0].replace('<b>', ''), i.split('</b>:')[1]] for i in x.split('<br>') if "</b>:" in i ])
#     print(df)

    def get_text_role(sent_list, role):
        '''
        Extract the utterances from the given role
        '''
        ans = ""
        for i in sent_list:
            ## if i[0].split(' ')[0] == role and i[0] != 'Phoebe Sr.':
            if i[0] == role:
                ans = ans + ' ' + i[1]
        return ans

    def get_context_role(sent_list, role):
        '''
        Extract the utterances not from the given role
        '''
        ans = ""
        for i in sent_list:
            # if i[0].split(' ')[0] != role or i[0] == 'Phoebe Sr.':
            if i[0] != role:
                ans = ans + ' ' + i[1]
        return ans
    def get_seg_id(sent_list, role):
        '''
        Generate the segment id for the whole sent
        '''
        ans = []
        for i in sent_list:
            if i[0].split(' ')[0] != role.split(' ')[0]:
                ans.append(0)
            else:
                ans.append(1)
        return ans

    def get_sent(sent_list, role):
        '''
        Obtain the whole sent
        '''
        ans = ""
        for i in sent_list:
            ans = ans + i[1]
        return ans
    df['dialog_state'] = df.apply(lambda r: get_seg_id(r['raw_text'], r['characters']), axis=1)
    df['sent'] = df.apply(lambda r: get_sent(r['raw_text'], r['characters']), axis=1)
    df['utterance'] = df.apply(lambda r: get_text_role(r['raw_text'], r['characters']), axis=1)
    df['context'] = df.apply(lambda r: get_context_role(r['raw_text'], r['characters']), axis=1)
    df['cAGR'] = df['cAGR'].apply(lambda x: 1 if x is True else 0)
    df['cCON'] = df['cCON'].apply(lambda x: 1 if x is True else 0)
    df['cEXT'] = df['cEXT'].apply(lambda x: 1 if x is True else 0)
    df['cOPN'] = df['cOPN'].apply(lambda x: 1 if x is True else 0)
    df['cNEU'] = df['cNEU'].apply(lambda x: 1 if x is True else 0)
    df['utterance'] = df['utterance'].apply(remove_link)
    return df

# process essay dataset
def essay_process():
    eassy = pd.read_csv('Source_Data/essays.csv',sep = ',')
    essay = eassy[['TEXT','cEXT','cNEU','cAGR','cCON','cOPN']]
    df_essay = essay.rename(columns={'TEXT':'utterance'})
    df_essay['cAGR'] = df_essay['cAGR'].apply(lambda x: 1 if x == 'y' else 0)
    df_essay['cCON'] = df_essay['cCON'].apply(lambda x: 1 if x == 'y' else 0)
    df_essay['cEXT'] = df_essay['cEXT'].apply(lambda x: 1 if x == 'y' else 0)
    df_essay['cOPN'] = df_essay['cOPN'].apply(lambda x: 1 if x == 'y' else 0)
    df_essay['cNEU'] = df_essay['cNEU'].apply(lambda x: 1 if x == 'y' else 0)
    df_essay['utterance'] = df_essay['utterance'].apply(remove_link)
    return df_essay

# tsv output
def tsv_generate(df,path,datasetName):
    for index, row in df.iteritems():
        if index == 'cEXT':
            df_E = df[['utterance','cEXT']]
            df_E.rename(columns={'cEXT':'labels'},inplace=True)
            df_E.to_csv(path + '/' + datasetName + '_E_whole.tsv',sep = '\t',index=False)
        elif index == 'cNEU':
            df_N = df[['utterance','cNEU']]
            df_N.rename(columns={'cNEU':'labels'},inplace=True)
            df_N.to_csv(path + '/' + datasetName + '_N_whole.tsv',sep = '\t',index=False)
        elif index == 'cAGR':
            df_A = df[['utterance','cAGR']]
            df_A.rename(columns={'cAGR':'labels'},inplace=True)
            df_A.to_csv(path + '/' + datasetName + '_A_whole.tsv',sep = '\t',index=False)
        elif index == 'cCON':
            df_C = df[['utterance','cCON']]
            df_C.rename(columns={'cCON':'labels'},inplace=True)
            df_C.to_csv(path + '/' + datasetName + '_C_whole.tsv',sep = '\t',index=False)
        elif index == 'cOPN':
            df_O = df[['utterance','cOPN']]
            df_O.rename(columns={'cOPN':'labels'},inplace=True)
            df_O.to_csv(path + '/' + datasetName + '_O_whole.tsv',sep = '\t',index=False)
            
# clean links
def remove_link(sentence):
    # removing links from text data
    sentence=re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+',' ',sentence)
    return sentence

def main():
    # get dataframe
    df_mp = myPersonality_process()
    df_pan = pan_process()
    df_friends = friends_process()
    df_essay = essay_process()
    # generate tsv
    tsv_generate(df_mp, 'myPersonality', 'MyPersonality')
    tsv_generate(df_pan, 'pan2015', 'Pan')
    tsv_generate(df_friends, 'FriendsPersona', 'Friends')
    tsv_generate(df_essay, 'Essay', 'Essay')
                
main()             