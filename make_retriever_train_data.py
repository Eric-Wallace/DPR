import json
import os

def read_json(filename):
    assert os.path.exists(filename), f"File {filename} not found"
    print(f"Reading {filename}")
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def write_json(stuff, filename):
    print(f"Writing {filename}")
    with open(filename, "w") as f:
        data = json.dumps(stuff, indent=4)
        f.write(data)

IN_FILE = "./smallout.json"
TRAIN_FILE = "./data/retriever_train.json"
VALID_FILE = "./data/retriever_valid.json"

data = read_json(IN_FILE)
results = []
i = 0
for entry in data:
    question = entry["question"]
    answer = entry["answers"]
    ctxs = entry["ctxs"]
    pos_ctx = None
    neg_ctx = None
#    print(len(ctxs))
 #   print([ctx["has_answer"] for ctx in ctxs])
    for ctx in ctxs:
  #      print(ctx)
        if ctx["has_answer"] and pos_ctx == None:
   #         print("changed answer")
            pos_ctx = [{"title": ctx["title"], "text": ctx["text"]}]
        if not ctx["has_answer"] and neg_ctx == None:
    #        print("changed neg answer")
            neg_ctx = [{"title": ctx["title"], "text": ctx["text"]}]
    if pos_ctx:
        i += 1
    if pos_ctx and neg_ctx:
        results.append(
            {"question": question,
                "answer": answer,
                "positive_ctxs": pos_ctx,
                "negative_ctxs": neg_ctx,
                "hard_negative_ctxs": []})

print(f"Got {len(data)} items from {IN_FILE}. \n Yielded {len(results)} items to {TRAIN_FILE} and {VALID_FILE}")
print(f"Got {i} has_answer out of {len(data)}")
train, valid = results[:-1000], results[-1000:]
write_json(train, TRAIN_FILE)
write_json(valid, VALID_FILE)
