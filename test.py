# with open('data/data/retriever/qas/nyt_mon_wed_2020_2021.tsv', 'r') as f:
#import mobypy
from tqdm import tqdm

#with open('google-10000-english/google-10000-english-usa-no-swears.txt', 'r') as f:
#    common = f.readlines()
#    common = set([c.strip().lower() for c in common][0:1000])
#common = []
#with open('/home/ericwallace/albertxu/xword/heuristic-data-collection/dictionary/opted_dict.tsv', 'r') as f:
#    for line in f:
#        common.append(line.split('\t')[1].strip().lower())
#common = set(common)

with open('../../albertxu/xword/data/closedbook-roberta-alldata-segment-acpt/train/train.tsv','r') as f:#data/data/retriever/qas/nyt_mon_wed_2020_2021.tsv', 'r') as f:
    lines = []
    for line in tqdm(f):
        q = line.split('\t')[0] 
        a = str(line.split('\t')[1].strip())
        #if len(q.split(' ')) == 1:
        #    continue#print('continuing len', line)
        if '\'' in a or '"' in a:
            continue
        #elif 'wordplay' in q or '?' in q or 'in a way' in q or 'perhaps' in q or 'slangily' in q:
        #    continue#pass#print('continuing wordplay', line)
        #elif any([i.lower() in common for i in a.split(' ')]):
        #    continue#print('continuing common', line)
        #elif len(mobypy.synonyms(a.lower())) > 0:
        #    continue#print('continuing in moby', line)
        #elif a.lower() in common:
        #    continue
        #else:
        lines.append(q + '\t' + '[\'' + a + '\']')
with open('train1m.tsv', 'w') as f:
    for i, line in enumerate(lines):
        if i == 1000000:
            break
        f.write(line + '\n')
