from tqdm import tqdm

common = []
with open('/home/ericwallace/albertxu/xword/heuristic-data-collection/dictionary/opted_dict.tsv', 'r') as f:
    for line in f:
        common.append(line.split('\t')[1].strip().lower())
common = set(common)

with open('../../albertxu/xword/data/closedbook-roberta-alldata-segment-acpt/train/train.tsv','r') as f:
    lines = []
    for line in tqdm(f):
        q = line.split('\t')[0] 
        a = str(line.split('\t')[1].strip())
        if len(q.split(' ')) == 1:
            continue
        if '\'' in a or '"' in a:
            continue
        elif 'wordplay' in q or '?' in q or 'in a way' in q or 'perhaps' in q or 'slangily' in q:
            continue
        elif a.lower() in common:
            continue
        lines.append(q + '\t' + '[\'' + a + '\']')
with open('train6m_pruned.tsv', 'w') as f:
    for i, line in enumerate(lines):
        f.write(line + '\n')
