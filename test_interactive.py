from interactive_reader import setup_reader, answer_clue
from interactive_openbook import setup_dpr

# setup_reader('data/hf_bert_base.cp')
setup_reader('test/dpr_reader.7.3608')
#[print("CHANGE BACK PKL !!") for i in range(100)]
setup_dpr("data/embeddings/downloads/checkpoint/retriever/single-adv-hn/nq/bert-base-encoder.cp", "data/wikipedia_split/psgs_w100.tsv", "data/embeddings/downloads/data/retriever_results/nq/single-adv-hn/wikipedia_passages_*.pkl", save_or_load_index=True)

#docs = get_docs("Niels Hydrogen Atom", 100, 4)
import pdb; pdb.set_trace()
answer_clue('Nucleic acid', 100)
#answer_clue('Qaboos bin Said\'s land', 100)
#answer_clue('Dwarf planet discovered in 2003', 100)
#answer_clue('___ Syed, subject of the 2014 "Serial" podcast', 100)
#answer_clue('Atomic structure theorist Niels', 100)
