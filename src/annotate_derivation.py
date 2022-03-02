import delphin
from delphin.codecs import simplemrs, dmrsjson
from delphin import itsdb, util

import argparse
import sys
import re
from pprint import pprint
from collections import defaultdict, Counter, deque
import json
import os


try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm
    
import networkx as nx
from networkx.algorithms.components import is_weakly_connected
    
from src import dg_util
import src.util as util

test_dir = []
alsoTest_dir = []
dev_dir = []
train_dir = []
bad_dir = []
virtual_dir = []

anno_config = dict()
config_name = None

def merge_trees(cleansed_sntid2item): 
    # return cleansed_sntid2item with derivation merged with syntactic tree's categories
    
    def sync_dfs_trees(deriv_nxDG, curr_deriv_node, syn_tree_nxDG, curr_syn_tree_node):
        if not syn_tree_nxDG[curr_syn_tree_node]:
            deriv_nxDG.nodes[curr_deriv_node]['cat'] = "<terminal>"
        else:
            deriv_nxDG.nodes[curr_deriv_node]['cat'] = syn_tree_nxDG.nodes[curr_syn_tree_node]['label']
#             print (deriv_nxDG.nodes[curr_deriv_node]['entity'], deriv_nxDG.nodes[curr_deriv_node]['cat'])
        curr_deriv_node_daughters = list(deriv_nxDG[curr_deriv_node])
        curr_syn_tree_node_daughters = list(syn_tree_nxDG[curr_syn_tree_node])
        if len(curr_deriv_node_daughters) == 2 and len(curr_syn_tree_node_daughters) == 2:
#             print ("!!")
            curr_deriv_node_left_daughter = [targ
                                             for src, targ, lbl in deriv_nxDG.out_edges(
                                                 nbunch = [curr_deriv_node],
                                                 data = 'label')
                                             if lbl == 'L'][0]
            curr_deriv_node_right_daughter = [targ
                                              for src, targ, lbl in deriv_nxDG.out_edges(
                                                  nbunch = [curr_deriv_node],
                                                  data = 'label')
                                              if lbl == 'R'][0]
            curr_syn_tree_node_left_daughter = [targ
                                                for src, targ, lbl in syn_tree_nxDG.out_edges(
                                                    nbunch = [curr_syn_tree_node],
                                                    data = 'label')
                                                if lbl == 'L'][0]
            curr_syn_tree_node_right_daughter = [targ
                                                 for src, targ, lbl in syn_tree_nxDG.out_edges(
                                                     nbunch = [curr_syn_tree_node],
                                                     data = 'label')
                                                 if lbl == 'R'][0]
            sync_dfs_trees(deriv_nxDG, curr_deriv_node_left_daughter, syn_tree_nxDG,
                           curr_syn_tree_node_left_daughter) 
            sync_dfs_trees(deriv_nxDG, curr_deriv_node_right_daughter, syn_tree_nxDG,
                           curr_syn_tree_node_right_daughter) 
        elif len(curr_deriv_node_daughters) == 1 and len(curr_syn_tree_node_daughters) == 1:
#             print ("!")
            curr_deriv_node_only_daughter = [targ
                                             for src, targ, lbl in deriv_nxDG.out_edges(
                                                 nbunch = [curr_deriv_node],
                                                 data = 'label')
                                             if lbl == 'U'][0]
            curr_syn_tree_node_only_daughter = [targ
                                                for src, targ, lbl in syn_tree_nxDG.out_edges(
                                                    nbunch = [curr_syn_tree_node],
                                                    data = 'label')
                                                if lbl == 'U'][0]
            sync_dfs_trees(deriv_nxDG, curr_deriv_node_only_daughter, syn_tree_nxDG,
                           curr_syn_tree_node_only_daughter)
        
            
    
    for snt_id in tqdm(cleansed_sntid2item):
        cleansed_item = cleansed_sntid2item[snt_id]
        deriv_nxDG = nx.readwrite.json_graph.node_link_graph(cleansed_item['derivation_nodelinkdict'])
        syn_tree_nxDG = nx.readwrite.json_graph.node_link_graph(cleansed_item['syn_tree_nodelinkdict'])
        curr_deriv_node = deriv_nxDG.graph['root']
#         if config_name == 'redwoods':
#             curr_deriv_node = list(deriv_nxDG.out_edges(curr_deriv_node))[0][1]
        curr_syn_tree_node = syn_tree_nxDG.graph['root']
#         print(deriv_nxDG.nodes[curr_deriv_node])
#         print(syn_tree_nxDG.nodes[curr_syn_tree_node])
        
#         util.write_figs_err(None, deriv_nxDG, None, "merge")
#         util.write_figs_err(None, syn_tree_nxDG, None, "merge")
#         input()
        sync_dfs_trees(deriv_nxDG, curr_deriv_node, syn_tree_nxDG, curr_syn_tree_node)
        cleansed_sntid2item[snt_id]['derivation_nodelinkdict'] = nx.readwrite.json_graph.node_link_data(deriv_nxDG)
    
    return cleansed_sntid2item
    
def annotate_data(cleansed_sntid2item):
    
    def remove_punc_rule(deriv_nxDG):
        anno_deriv_nxDG = deriv_nxDG.copy()
        for node, node_prop in deriv_nxDG.nodes(data = True):
            if 'entity' in node_prop and node_prop['entity'].endswith("_plr"):
                prev_node = list(anno_deriv_nxDG.in_edges(node))[0][0]
                next_nodes = list(anno_deriv_nxDG.out_edges(node))[0]
                edge_label = anno_deriv_nxDG.edges[(prev_node, node)]['label']
                if next_nodes:
                    for next_node in next_nodes:
                        anno_deriv_nxDG.add_edge(prev_node, next_node, label = edge_label)
                anno_deriv_nxDG.remove_node(node)
        return anno_deriv_nxDG
    
    def lexical_collapsing(deriv_nxDG):
        anno_deriv_nxDG = deriv_nxDG.copy()
#         print (deriv_nxDG.nodes)
#         print ()
#         print (deriv_nxDG.out_edges())
#         print ()
        for node, node_prop in deriv_nxDG.nodes(data = True):
            # if terminal
            if not deriv_nxDG[node]:
                preterm_node = list(deriv_nxDG.in_edges(node))[0][0]
                collapsed_preterm_symbol = ""
                prev_node = node
                curr_node = preterm_node
                highest_lex_node = preterm_node
                # while subtree rooted at curr_node is still a unary chain and curr_node is not root_*
                while(len(deriv_nxDG[curr_node]) == 1 and len(deriv_nxDG.in_edges(nbunch = [curr_node])) == 1):
                    # if not syntactic rule
                    
                    if not util.is_syn_construction(deriv_nxDG.nodes[curr_node]['entity']):
                        if not collapsed_preterm_symbol == "":
                            collapsed_preterm_symbol += "/"
                        collapsed_preterm_symbol = collapsed_preterm_symbol + deriv_nxDG.nodes[curr_node]['entity']
                        
                        # check if any lexical rule comes after syntactic rule
                        if 'entity' in deriv_nxDG.nodes[prev_node]\
                            and util.is_syn_construction(deriv_nxDG.nodes[curr_node]['entity']):
                            print (snt_id, ": _c after deriv. rule! Only possible for pre-root node")
                            print (deriv_nxDG.nodes[list(deriv_nxDG.in_edges(curr_node))[0][0]]['entity'])
                            
                    next_node = list(deriv_nxDG.in_edges(curr_node))[0][0]
                    # if parent of current node is still unary and both parent and current node are derivational/lexical
                    if len(deriv_nxDG[next_node]) == 1\
                        and not util.is_syn_construction(deriv_nxDG.nodes[curr_node]['entity'])\
                        and not util.is_syn_construction(deriv_nxDG.nodes[next_node]['entity']):
                        anno_deriv_nxDG.remove_node(curr_node)
                        highest_lex_node = next_node
                    prev_node = curr_node
                    curr_node = next_node
                
                
                if highest_lex_node != node:
                    anno_deriv_nxDG.add_edge(highest_lex_node, node, label = 'U')
                    assert len(anno_deriv_nxDG[highest_lex_node]) == 1
                anno_deriv_nxDG.nodes[highest_lex_node]['entity'] = collapsed_preterm_symbol

        # print (len(anno_deriv_nxDG.nodes))
        # pprint ([(node,anno_deriv_nxDG.nodes[node]['entity']) for node in anno_deriv_nxDG.nodes if 'entity' in anno_deriv_nxDG.nodes[node]])
        # print (len(anno_deriv_nxDG.out_edges()))
        if len(anno_deriv_nxDG.nodes) - 1 != len(anno_deriv_nxDG.out_edges()):
            print ("node != edge + 1")
            print (len(anno_deriv_nxDG.out_edges()))
            print (len(anno_deriv_nxDG.nodes))
        return anno_deriv_nxDG
    
    def vertical_markovization(deriv_nxDG, conf):
        def propagate_parents(anno_deriv_nxDG, curr_node, q, halfMarkov = False, half = False, halfhalf = False):
            # curr_node is terminal => return
            if not anno_deriv_nxDG[curr_node]:
                return
            # syn_only, then do not annotate any parent for lexical rule (collapsed or not)
            if syn_only:
                if all([not anno_deriv_nxDG[node] for node in anno_deriv_nxDG[curr_node]]):
                    return 
            ancestors_in_order = q.copy()
            ancestors_in_order.reverse()
            q.popleft()
            q.append(anno_deriv_nxDG.nodes[curr_node]['entity'])
            if halfMarkov:
                ancestor = ancestors_in_order[0]
                if not ancestor[:4] == 'root':
                    ancestor =  ancestor.split("_")[0] + "_" + ancestor.split("_")[-1]
                anno_deriv_nxDG.nodes[curr_node]['entity'] += "^" + ancestor
            elif half:
                if not anno_deriv_nxDG.nodes[curr_node]['entity'] == 'root':
                    anno_deriv_nxDG.nodes[curr_node]['entity'] = anno_deriv_nxDG.nodes[curr_node]['entity'].split("_")[0] + '_'\
                        + anno_deriv_nxDG.nodes[curr_node]['entity'].split("_")[-1]
            elif halfhalf:
                ancestor = ancestors_in_order[0]
                if not ancestor[:4] == 'root':
                    ancestor =  ancestor.split("_")[0] + "_" + ancestor.split("_")[-1]
                if anno_deriv_nxDG.nodes[curr_node]['entity'] != 'root':
                    anno_deriv_nxDG.nodes[curr_node]['entity'] = anno_deriv_nxDG.nodes[curr_node]['entity'].split("_")[0]\
                        + "_" + anno_deriv_nxDG.nodes[curr_node]['entity'].split("_")[-1] + "^" + ancestor
                else:
                    anno_deriv_nxDG.nodes[curr_node]['entity'] = anno_deriv_nxDG.nodes[curr_node]['entity'] + "^" + ancestor
            else:
                anno_deriv_nxDG.nodes[curr_node]['entity'] += "^" + "^".join([ancestor for ancestor in ancestors_in_order])
            daughters = list(anno_deriv_nxDG[curr_node])
            for daughter in daughters:
                temp_q = q.copy()
                propagate_parents(anno_deriv_nxDG, daughter, temp_q, halfMarkov, half, halfhalf)
        
        order, syn_only = conf["order"], conf["syntactic_only"]
        if order == 1:
            return deriv_nxDG
        anno_deriv_nxDG = deriv_nxDG.copy()
        root = anno_deriv_nxDG.graph['root']
        q = deque() 
        halfMarkov = False
        half = False
        halfhalf = False
        if order == 0.5:
            order = 1
            half = True
        if order == 1.5:
            order = 2
            halfMarkov = True
        if order == 1.01:
            order = 2
            halfhalf = True
        for i in range(order - 2):
            q.append("")
        q.append(deriv_nxDG.nodes[list(deriv_nxDG.in_edges(root))[0][0]]['entity'])
        propagate_parents(anno_deriv_nxDG, root, q, halfMarkov, half, halfhalf)
        return anno_deriv_nxDG
    
    def add_syn_cat(deriv_nxDG, conf):
        anno_deriv_nxDG = deriv_nxDG.copy()
        apply, syn_only, cat_only, coarse = conf["apply"], conf["syntactic_only"], conf.get('cat_only'), conf.get('coarse')
        if not apply:
            return anno_deriv_nxDG
        for node, node_prop in anno_deriv_nxDG.nodes(data = True):
            if 'cat' in anno_deriv_nxDG.nodes[node] and anno_deriv_nxDG.nodes[node]['cat'] != "<terminal>":
#                 print (anno_deriv_nxDG.nodes[node])
                cat = anno_deriv_nxDG.nodes[node]['cat']
                if coarse: cat = cat.split("/")[0]
                if syn_only:
                    real_entity = anno_deriv_nxDG.nodes[node]['entity'].split("^")[0]
                    if util.is_syn_construction(real_entity):
                        if not cat_only:
                            if not util.is_derivNode_preTerm(anno_deriv_nxDG, node):
                                anno_deriv_nxDG.nodes[node]['entity'] += "&" + cat
                        else:
                            if not util.is_derivNode_preTerm(anno_deriv_nxDG, node):
                                anno_deriv_nxDG.nodes[node]['entity'] = cat
    #                         print (anno_deriv_nxDG.nodes[node]['entity'])
                else:
                    if not cat_only:
                        anno_deriv_nxDG.nodes[node]['entity'] += "&" + cat
                    else:
                        if not is_derivNode_preTerm(anno_deriv_nxDG, node):
                            anno_deriv_nxDG.nodes[node]['entity'] = cat
        return anno_deriv_nxDG
    training_sntid2item = defaultdict(dict)
    dev_sntid2item = defaultdict(dict)
    test_sntid2item = defaultdict(dict)
    
    sys.stderr.write('Merging derivations and syntactic trees ...\n')
    merged_sntid2item = merge_trees(cleansed_sntid2item)
    
    sys.stderr.write('Annotating derivations trees ...\n')
    for snt_id in tqdm(merged_sntid2item):
        bad_item = False
#         if snt_id != "wsj18c/21849020":
#             continue
        merged_item = merged_sntid2item[snt_id]
        
        anno_deriv_nxDG = nx.readwrite.json_graph.node_link_graph(merged_item['derivation_nodelinkdict'])
        
        if anno_config["remove_punc_rule"]:
            anno_deriv_nxDG = remove_punc_rule(anno_deriv_nxDG)
        if anno_config["lexical_collapsing"]:
            anno_deriv_nxDG = lexical_collapsing(anno_deriv_nxDG)
        if anno_config["unary_syntactic_rule_collapsing"]:
            # not implemented
            anno_deriv_nxDG = anno_deriv_nxDG
        anno_deriv_nxDG = vertical_markovization(anno_deriv_nxDG, anno_config["vertical_markovization"])
        
        # approximate head feature with syntactic category given in syntactic tree
        anno_deriv_nxDG = add_syn_cat(anno_deriv_nxDG, anno_config["syntactic_cat_feature"])
        
        merged_item['anno_derivation_nodelinkdict'] = nx.readwrite.json_graph.node_link_data(anno_deriv_nxDG)
        profile = snt_id.split("/")[0]
        if profile in dev_dir:
            dev_sntid2item[snt_id] = merged_item
        elif profile in test_dir or profile in alsoTest_dir:
            test_sntid2item[snt_id] = merged_item
        else:
            training_sntid2item[snt_id] = merged_item
        
    return training_sntid2item, dev_sntid2item, test_sntid2item
        
def main(preprocessedData_dir, annotatedData_dir, config_filepath, sampleOrAll):

    global test_dir, alsoTest_dir, dev_dir, train_dir, bad_dir, virtual_dir, anno_config
    global config_name
    sys.stderr.write("Loading config file ...\n")
    with open(config_filepath, "r", encoding='utf-8') as f:
        config = json.load(f)
    
    config_name = config_filepath.split("_")[1].split("-")[0]
    test_dir = config['data_split']['test_profiles']
    alsoTest_dir = config['data_split']['also_test_profiles']
    dev_dir = config['data_split']['dev_profiles']
    train_dir = config['data_split']['training_profiles']
    bad_dir = config['data_split']['bad_profiles']
    virtual_dir = config['data_split']['virtual_dirs']
    anno_config = config['annotation']
                
    all_cleansed_data_filepath = os.path.join(preprocessedData_dir, "all", "cleansed_sntid2item.json")
    sample_cleansed_data_filepath = os.path.join(preprocessedData_dir,  "all", "sample_cleansed_sntid2item.json")
    if sampleOrAll == 'sample':
        cleansed_data_filepath = sample_cleansed_data_filepath
    elif sampleOrAll == 'all':
        cleansed_data_filepath = all_cleansed_data_filepath
    
    all_training_data_filepath = os.path.join(annotatedData_dir, "training", "training_sntid2item.json")
    all_dev_data_filepath = os.path.join(annotatedData_dir, "dev", "dev_sntid2item.json")
    all_test_data_filepath = os.path.join(annotatedData_dir, "test", "test_sntid2item.json")
    sample_training_data_filepath = os.path.join(annotatedData_dir, "training", "sample_training_sntid2item.json")
    sample_dev_data_filepath = os.path.join(annotatedData_dir, "dev", "sample_dev_sntid2item.json")
    sample_test_data_filepath = os.path.join(annotatedData_dir, "test", "sample_test_sntid2item.json")
    if sampleOrAll == 'sample':
        training_data_filepath = sample_training_data_filepath
        dev_data_filepath = sample_dev_data_filepath
        test_data_filepath = sample_test_data_filepath
    elif sampleOrAll == 'all':
        training_data_filepath = all_training_data_filepath
        dev_data_filepath = all_dev_data_filepath
        test_data_filepath = all_test_data_filepath

    # merge syntactic tree and derivation and annotate derivation
    sys.stderr.write("Loading cleansed data ...\n")
    with open(cleansed_data_filepath, "r", encoding='utf-8') as f:
        cleansed_sntid2item = json.load(f)
    sys.stderr.write("Loaded!\n")
    sys.stderr.write('Annotating derivation to approximate HPSG with CFG ...\n')
    train_sntid2item, dev_sntid2item, test_sntid2item = annotate_data(cleansed_sntid2item)
    sys.stderr.write("Writing annotated data ...\n")
    os.makedirs(training_data_filepath.rsplit("/", 1)[0], exist_ok=True)
    with open(training_data_filepath, "w", encoding='utf-8') as f:
        json.dump(train_sntid2item, f)#, indent=2
    os.makedirs(dev_data_filepath.rsplit("/", 1)[0], exist_ok=True)
    with open(dev_data_filepath, "w", encoding='utf-8') as f:
        json.dump(dev_sntid2item, f)#, indent=2
    os.makedirs(test_data_filepath.rsplit("/", 1)[0], exist_ok=True)
    with open(test_data_filepath, "w", encoding='utf-8') as f:
        json.dump(test_sntid2item, f) #, indent=2
    sys.stderr.write('Done! Written annotated data to files!\n')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('preprocessedData_dir', help='path to preprocessed data directory')
    parser.add_argument('config_filepath', help='path to config file')
    parser.add_argument('sampleOrAll', default='all', help='try with sample data only or not')
    args = parser.parse_args()
    main(args.preprocessedData_dir, args.annotatedData_dir, args.config_filepath, args.sampleOrAll)

