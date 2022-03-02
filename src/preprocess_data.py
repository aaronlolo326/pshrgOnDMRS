import delphin
from delphin.codecs import simplemrs, dmrsjson
from delphin import itsdb

import argparse
import sys
import re
from pprint import pprint
from collections import defaultdict, Counter
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

# test_dir = ["cb","ecpr","jhk","jhu","tgk","tgu","psk","psu","rondane","vm32","ws13","petet"]
# alsoTest_dir = "cf04 cf06 cf10 cf21 cg07 cg11 cg21 cg25 cg32 cg35 ck11 ck17 cl05 cl14 cm04 cn03 cn10 cp15 cp26 cr09 ws214 wsj21a wsj21b wsj21c wsj21d".split(" ")
# dev_dir = "ecpa jh5 tg2 ws12 wsj20a wsj20b wsj20c wsj20d wsj20e".split(" ")
# train_dir = ["(the rest)"]
# bad_dir = ["pest"]
# virtual_dir = ['.svn','redwoods','wsj21','wescience','brown']

test_dir = []
alsoTest_dir = []
dev_dir = []
training_dir = []
bad_dir = []
virtual_dir = []
sample_dir = ["esd","jh5","psk"]


def organize_to_dict(redwoods_dir = None, gigaword_parsed_path = None, sampleOrAll = "sample", redwdOrGw = "redwoords", for_aceGen = False):
    sntid2item = defaultdict(dict)
    filt_sntid2item = defaultdict(dict)
    sntid2cnt = Counter()
    amb_sntid = set()
    bad_mrs_sntid = set()
    bad_syn_tree_sntid = set()
    bad_deriv_sntid = set()
    no_parse_sntid = set()
    unusable_sntid = set()
    usable_sntid = set()
    filt_trainingDevTest_cnt = [0,0,0]
    orig_trainingDevTest_cnt = [0,0,0]
    
    
    if redwdOrGw == 'redwoods':
        '''
        Organize data to dict
        Retain ONE parse for multiple parses (ambiguous)
        '''
        for directory in tqdm(os.listdir(redwoods_dir)):
            if directory in virtual_dir or directory in bad_dir:
                continue    
            dir_path = os.path.join(redwoods_dir,directory)
            ts = itsdb.TestSuite(dir_path)

            # if generate sample for testing
            if sampleOrAll == 'sample':
                if not directory in sample_dir:
                    continue
            if for_aceGen:
                if not (directory in test_dir or directory in alsoTest_dir):
                    continue
                else: orig_trainingDevTest_cnt[2] += len(ts['item'])
            else:
                if directory in dev_dir:
                    orig_trainingDevTest_cnt[1] += len(ts['item'])
                elif directory in test_dir or directory in alsoTest_dir:
                    orig_trainingDevTest_cnt[2] += len(ts['item'])
                else:
                    orig_trainingDevTest_cnt[0] += len(ts['item'])
            # add sentence
            for idx, item in enumerate(ts['item']):
                key = directory + "/" + str(item['i-id'])
                sntid2cnt[key] += 1
                if directory == 'pest':
                    sntid2item[key] = {"profile": directory, "sentence": item['i-comment']}
                else:
                    sntid2item[key] = {"profile": directory, "sentence": item['i-input']}
            # add syntactic tree, derivation and dmrs for each sentence
            for idx, result in enumerate(ts['result']):
                key = directory + "/" + str(result['parse-id'])
                if any([k in sntid2item[key] for k in ['mrs_str', 'syn_tree', 'derivation', 'dmrs_json']]):
                    amb_sntid.add(key)
                    pass 
                else:
                    try:
                        mrs_str = result['mrs']
                        mrs = simplemrs.loads(mrs_str)
                        dmrs_str = delphin.dmrs.from_mrs(mrs[0])
                        dmrs_json = json.loads(dmrsjson.encode(dmrs_str, indent=True))
                        sntid2item[key]['dmrs_json'] = dmrs_json
                        sntid2item[key]['mrs_str'] = mrs_str
                    except:
                        bad_mrs_sntid.add(key)
                    if not for_aceGen:
                        try:
                            sntid2item[key]['syn_tree'] = delphin.util.SExpr.parse(result['tree']).data
                        except Exception as e:
                            bad_syn_tree_sntid.add(key)
                        try:
                            sntid2item[key]['derivation'] = delphin.derivation.from_string(result['derivation'])
                        except Exception as e:
                            bad_deriv_sntid.add(key)
        badparse_sntid = bad_deriv_sntid.union(bad_syn_tree_sntid).union(bad_mrs_sntid)
        '''
        Report on data filtering results
        '''
        for key in sntid2item:
            if key in badparse_sntid or key in amb_sntid:
                continue
            if not for_aceGen:
                if any([k not in sntid2item[key] for k in ['syn_tree', 'derivation', 'dmrs_json']]):
                    no_parse_sntid.add(key)
                    continue
            else:
                if 'mrs_str' not in sntid2item[key] or 'dmrs_json' not in sntid2item[key]:
                    no_parse_sntid.add(key)
                    continue
            directory = sntid2item[key]['profile']
            if directory in dev_dir and not for_aceGen:
                filt_trainingDevTest_cnt[1] += 1
            elif directory in test_dir or directory in alsoTest_dir:
                filt_trainingDevTest_cnt[2] += 1
            elif not for_aceGen:
                filt_trainingDevTest_cnt[0] += 1
                
    elif redwdOrGw == "gigaword":
        assert not for_aceGen 
        directory = "gigaword"
        with open(gigaword_parsed_path, "r") as f:
            gigaword_id2data = json.load(f)
            
        for snt_id, item in enumerate(tqdm(gigaword_id2data)):
            if not item or not item[1]:
                continue
            snt, results = item
            key = directory + "/" + str(snt_id)
            sntid2item[key] = {"profile": directory, "sentence": snt}
            
            # if generate sample for testing
            if sampleOrAll == 'sample':
                if snt_id > 100:
                    break
            
            # add syntactic tree, derivation and dmrs for each sentence
            for idx, result in enumerate(results):
                try:
                    mrs_str = result['mrs']
                    mrs = simplemrs.loads(mrs_str)
                    dmrs_str = delphin.dmrs.from_mrs(mrs[0])
                    sntid2item[key]['dmrs_json'] = json.loads(dmrsjson.encode(dmrs_str, indent=True))
                except:
                    bad_mrs_sntid.add(key)
                try:
                    sntid2item[key]['syn_tree'] = delphin.util.SExpr.parse(result['tree']).data
                except Exception as e:
                    bad_syn_tree_sntid.add(key)
                try:
                    sntid2item[key]['derivation'] = delphin.derivation.from_string(result['derivation'])
                except Exception as e:
                    bad_deriv_sntid.add(key)
        badparse_sntid = bad_deriv_sntid.union(bad_syn_tree_sntid).union(bad_mrs_sntid)
        '''
        Report on data filtering results
        '''
        for key in sntid2item:
            if key in badparse_sntid or key in amb_sntid:
                continue
            if ('syn_tree' or 'derivation' or 'dmrs_json') not in sntid2item[key]:
                no_parse_sntid.add(key)
                continue
            # is always training
            filt_trainingDevTest_cnt[0] += 1
                
                
                
    unusable_sntid = badparse_sntid.union(no_parse_sntid).union(amb_sntid)
    usable_sntid = set(sntid2item.keys()) - unusable_sntid
#     print (usable_sntid)
    
    if redwdOrGw == 'gigaword':
        sys.stderr.write('''originally, total # items: {}
    where (training,dev,test) = {}
    After filtering,
    # items with bad MRS: {} #775
    # else items with bad syn tree: {}
    # else items with bad deriv: {}
    # items unusable: {}
    # items with multiple parses (ambiguous): {}
    # usable (only one disambiguated parse and w/ good MRS): {}
    where (training,dev,test) = {}\n'''.format(len(gigaword_id2data),
                                          (1e6, 0 ,0),
                                          len(bad_mrs_sntid),
                                          len(bad_syn_tree_sntid),
                                          len(bad_deriv_sntid),
                                          len(unusable_sntid),
                                          len(amb_sntid),
                                          len(usable_sntid),
                                          filt_trainingDevTest_cnt
                                         )
                        )
        
    elif redwdOrGw == 'redwoods':
        sys.stderr.write('''originally, total # items: {}
    where (training,dev,test) = {}
    After filtering,
    # items with bad MRS: {} #775
    # else items with bad syn tree: {}
    # else items with bad deriv: {}
    # items unusable: {}
    # items with multiple parses (ambiguous): {}
    # usable (only one disambiguated parse and w/ good MRS): {}
    where (training,dev,test) = {}\n'''.format(len(orig_trainingDevTest_cnt),
                                          orig_trainingDevTest_cnt,
                                          len(bad_mrs_sntid),
                                          len(bad_syn_tree_sntid),
                                          len(bad_deriv_sntid),
                                          len(unusable_sntid),
                                          len(amb_sntid),
                                          len(usable_sntid),
                                          filt_trainingDevTest_cnt
                                         )
                        )
    '''
    Write to file
    Only those with exactly one parse and good MRS
    '''
    filt_sntid = usable_sntid
    for key in tqdm(sntid2item):
        if key in filt_sntid:
            
            erg_digraphs = dg_util.Erg_DiGraphs()
#             print (key)
            is_good_dmrs = erg_digraphs.init_dmrsjson(sntid2item[key]['dmrs_json'])
            if not is_good_dmrs:
                continue
            erg_digraphs.init_snt(sntid2item[key]['sentence'])
            filt_sntid2item[key]['sentence'] = erg_digraphs.snt
            if not for_aceGen:
                filt_sntid2item[key]['dmrs_nodelinkdict'] = nx.readwrite.json_graph.node_link_data(
                    erg_digraphs.dmrs_dg)
                erg_digraphs.init_syn_tree(sntid2item[key]['syn_tree'])
                erg_digraphs.init_erg_deriv(sntid2item[key]['derivation'])
                filt_sntid2item[key]['syn_tree_nodelinkdict'] = nx.readwrite.json_graph.node_link_data(
                    erg_digraphs.syn_tree_dg)
                filt_sntid2item[key]['derivation_nodelinkdict'] = nx.readwrite.json_graph.node_link_data(
                    erg_digraphs.deriv_dg)
            else:
                filt_sntid2item[key]['mrs_str'] = sntid2item[key]['mrs_str']
#             if key.endswith("0"):
#                 break
    sys.stderr.write("Filtered item count: {}\n".format(len(filt_sntid2item)))
    print ()
    return filt_sntid2item
    # return filt_sntid2item
    
def cleanse_data(filt_sntid2item, redwdOrGw):
    
    def normalize_sentence(sentence):
        """Make formatting consistent."""
        # remove wikipedia link brackets
        sentence = re.sub("(\[|\])", " ", sentence)
        # normalize double quotes
        sentence = re.sub("’|`", "'", sentence)
        # sentence = re.sub("''", '"', sentence)
        return sentence
    
    def is_drop_node(pred):
        drop_preds = set(["udef_q", "pronoun_q", "number_q", "proper_q", "def_explicit_q", "def_implicit_q"])
        return (pred in drop_preds)
    
    def drop_nodes(orig_dmrs_nxDG):
        if not is_weakly_connected(orig_dmrs_nxDG):
            return None
        cleansed_dmrs_nxDG = orig_dmrs_nxDG.copy()
        to_be_dropped_cnt = 0
        in_deg = False
        for node, node_prop in orig_dmrs_nxDG.nodes(data=True):
            pred = node_prop['instance']
            if is_drop_node(pred):
                cleansed_dmrs_nxDG_tmp = cleansed_dmrs_nxDG.copy()
                cleansed_dmrs_nxDG_tmp.remove_node(node)
                if is_weakly_connected(cleansed_dmrs_nxDG_tmp):
                    cleansed_dmrs_nxDG = cleansed_dmrs_nxDG_tmp
        return cleansed_dmrs_nxDG
    
    def change_unknown_pred_rep(pred):
        new_pred = re.sub(r'_([^\s]+)\/(.*?)_u_unknown', r'_\1_\2_unknown', pred)
        return new_pred
    
    def adjust_dmrs_anchor_boundaries(dmrs_nxDG, sentence):
        len_snt = len(sentence)
        for node, node_prop in dmrs_nxDG.nodes(data=True):
            try:
                anchor_from, anchor_to = util.anchorstr2anchors(node_prop['lnk'])
            except:
                return False
            orig_anchor_from, orig_anchor_to = (anchor_from, anchor_to)
            
            ## for unknown, count from start to len(<the unk word>) and keep this portion
            if node_prop['instance'].endswith("_unknown"):
                x = re.search(r'_(.+)_(.+)_unknown', node_prop['instance'])
                if x != None:
                    anchor_to = anchor_from + len(x.group(1))
                    dmrs_nxDG.nodes[node]['lnk'] = "<{}:{}>".format(anchor_from, anchor_to)
                    continue
                else:
                    print (node_prop['instance'], ": regex search err?!")
            ## for named, record the length of carg ad make sure the anchoring isnt trimmed too short
            len_carg = 1
            if node_prop['instance'].startswith("named"):
                len_carg = len(node_prop['carg'])
                
            # remove trailing punctuation
            ## remove anchor to "."/"?" if it's end of sentence. O/w keep, abbrev.
            if sentence[anchor_to-1] in [".", "?", ","] and anchor_to == len_snt:
                anchor_to -= 1
            while (anchor_from+1 < anchor_to) or\
                  ((anchor_to - anchor_from) > len_carg and node_prop['instance'].startswith("named")):
                try:
                    if not sentence[anchor_to-1].isalnum():
                        ## general case
                        if not sentence[anchor_to-1] in ["'", ".", ","]:
                            anchor_to -= 1
                        elif node_prop['instance'].startswith("named"):
                            break
                        ## e.g. flies' predator's human's. : flies(')-> can't remove; human('s.) -> remove "."
                        elif sentence[anchor_to-1] == "'" and not node_prop['instance'] == 'poss':
                            anchor_to -= 1
                        else:
                            break
                    else:
                        break
                except Exception as e:
                    return False
                
            # remove preceding punctuation
            while (anchor_from+1 < anchor_to and not node_prop['instance'].startswith("named")) or\
                  ((anchor_to - anchor_from) > len_carg and node_prop['instance'].startswith("named")):
                if not sentence[anchor_from].isalnum():
                    ## _would_v_modal ('d') -> remove nth
                    ## ('s) -> remove nth
                    ## <be> -> remove nth
                    if not sentence[anchor_from:anchor_from+2].lower() == "'s" or\
                       not node_prop['instance'] == '_would_v_modal' and sentence[anchor_from:anchor_from+2].lower() == "'d" or\
                       not node_prop['instance'].startswith('_be_') and not sentence[anchor_from:anchor_from+2].lower() in ["'s","'m"] and not sentence[anchor_from:anchor_from+3].lower() == "'re":
                        anchor_from += 1
                    else:
                        break
                else:
                    break
            dmrs_nxDG.nodes[node]['lnk'] = "<{}:{}>".format(anchor_from, anchor_to)
            # print (sentence[anchor_from:anchor_to])
        return True
    
#     def adjust_deriv_anchor_boundaries(deriv, sentence):
# #         apostrophe_s_12_lex has_aux_cx_3 be_c_is_cx_3 # 's
# #         apostrophe_s_13_lex # '
# #         w_period_plr # *.
# #         would_aux_pos_cx_3 had_aux_cx_3 # 'd
# #         have_fin_aux_cx_3 # 've
# #         be_c_am_cx_3 # 'm
# #         be_c_are_cx_2 # 're
# #         will_aux_pos_cx_3 #'ll
#         surface2puncRule = {"'s": set("apostrophe_s_12_lex", "has_aux_cx_3", "be_c_is_cx_3"),
#                             "'": set("apostrophe_s_13_lex")
#                             "'d": set("would_aux_pos_cx_3", "had_aux_cx_3"),
#                             "'ve": set("have_fin_aux_cx_3"),
#                             "'m": set("be_c_am_cx_3"),
#                             "'re": set("be_c_are_cx_2"),
#                             "'ll": set("will_aux_pos_cx_3"),
#                             "…": set("threedots_disc_adv4", "punct_ellipsis_r", "punct_ellipsis_r2")
#                            }
#         len_snt = len(sentence)
#         for node, node_prop in deriv.nodes(data=True):
#             if deriv.out_degree(node) == 0:
#                 # terminal node
#                 cont = False
#                 anchor_from, anchor_to = (node_prop['from'], node_prop['to'])
#                 orig_anchor_from, orig_anchor_to = (anchor_from, anchor_to)
#                 anchored_surface.lower() = sentence[anchor_from:anchor_to].lower()

#                 # remove trailing punctuation
#                 # remove anchor to "."/"?" if it's end of sentence. O/w keep, abbrev.
#                 if sentence[anchor_to-1] in [".", "?"] and anchor_to == len_snt:
#                     anchor_to -= 1
                
#                 curr_node = deriv.in_edges(curr_node)[0][0]
#                 while len(deriv.out_edges(curr_node)) == 1:
#                     if deriv.nodes[curr_node]['entity'] == "genericname" and
#                        deriv.nodes[curr_node]['entity'] == "generic_proper_ne":
#                         cont = True
#                         break                 
#                 if cont:
#                     continue
#                 if anchored_surface in surface2puncRule:
#                     continue
#                 # remove trailing punctuation
#                 # notice: flies' predator's human's. : flies(')-> can't remove; human('s.) -> remove .
#                 while anchor_from+1 < anchor_to:
#                     if not sentence[anchor_to-1].isalnum():
#                         ## general case
#                         if not sentence[anchor_to-1] in ["'", "."]:
#                             anchor_to -= 1
#                         else:
#                             curr_node = deriv.in_edges(curr_node)[0][0]
#                             while len(deriv.out_edges(curr_node)) == 1:
#                                 if deriv.nodes[curr_node]['entity'] == "w_period_plr"
#                                     anchor_to -= 1
#                                     break
#                                 else:
#                                     curr_node = deriv.in_edges(curr_node)[0][0]
#                         else:
#                             break
#                     # remove preceding punctuation
#                     # adjust start of replacement anchor to exclude punctuation
#                     # notice: you'd be: 'd -> can't remove?
#                         # check pred, e.g.:
#                             # _would_v_modal ('d') -> remove nth
#                             # poss ('s) -> remove nth
#                     while anchor_from+1 < anchor_to:
#                         if not sentence[anchor_from].isalnum():
#                             if not node_prop['instance'] == 'poss' and sentence[anchor_from:anchor_from+2] == "'s" or\
#                                not node_prop['instance'] == '_would_v_modal' and sentence[anchor_from:anchor_from+2] == "'d" or\
#                                not node_prop['instance'].startswith('_be_') and not sentence[anchor_from:anchor_from+2] in ["'s","'m"] and not sentence[anchor_from:anchor_from+3] == "'re":
#                                 anchor_from += 1
#                             else:
#                                 break
#                         else:
#                             break

#         #         root = deriv.out_edges(nbunch = [None])[0][1]
    
    def propagate_anchors(deriv, curr_node):
        noOfChildren = len(deriv.out_edges(nbunch = [curr_node]))

        # Terminal
        if noOfChildren == 0:
            # normalize form
            deriv.nodes[curr_node]['form'] = normalize_sentence(deriv.nodes[curr_node]['form'])
            return (deriv.nodes[curr_node]['anchor_from'], deriv.nodes[curr_node]['anchor_to'])
        # Non-terminal
        else:
            children_anchors = [propagate_anchors(deriv, list(deriv.out_edges(nbunch = [curr_node]))[i][1])
                     for i in range(noOfChildren)]
            deriv.nodes[curr_node]['anchor_from'] = str(min([int(ancfrom) for ancfrom, ancto in children_anchors]))
            deriv.nodes[curr_node]['anchor_to'] = str(max([int(ancto) for ancfrom, ancto in children_anchors]))
#             print (deriv.nodes[curr_node]['entity'], children_anchors,
#                   deriv.nodes[curr_node]['anchor_from'], deriv.nodes[curr_node]['anchor_to'])
            return (deriv.nodes[curr_node]['anchor_from'], deriv.nodes[curr_node]['anchor_to'])
        
    def syn_tree_anchors_from_deriv(syn_tree, curr_syn_tree_node, deriv, curr_deriv_node):
        noOfChildren = len(syn_tree.out_edges(nbunch = [curr_syn_tree_node]))
        syn_tree.nodes[curr_syn_tree_node]['anchor_from'] = deriv.nodes[curr_deriv_node]['anchor_from']
        syn_tree.nodes[curr_syn_tree_node]['anchor_to'] = deriv.nodes[curr_deriv_node]['anchor_to']
        if noOfChildren > 0:
            for idx, (src,targ) in enumerate(list(syn_tree.out_edges(nbunch = [curr_syn_tree_node]))):
                try:
                    next_deriv_node = list(deriv.out_edges(nbunch = [curr_deriv_node]))[idx][1]
                    syn_tree_anchors_from_deriv(syn_tree, targ, deriv, next_deriv_node)
                except:
                    # print (idx, deriv.out_edges(nbunch = [curr_deriv_node]))
                    return False
        return True
        

    
    cleansed_sntid2item = defaultdict(dict)
    
    prop_err_cnt = 0
    copy_err_cnt = 0
    
    for snt_id in tqdm(filt_sntid2item):
        bad_item = False
#         if snt_id != "wsj18c/21847040":
#             continue
        filt_item = filt_sntid2item[snt_id]
        
        ### normalize sentence ###
        cleansed_sntid2item[snt_id]['sentence'] = normalize_sentence(filt_item['sentence'])
        len_snt = len(cleansed_sntid2item[snt_id]['sentence'])
#         print ()
#         print (snt_id)
#         print (cleansed_sntid2item[snt_id]['sentence'])

        ### cleanse DMRS ###
        # delete useless nodes and their incident edges
        # if disconnected, recover the node
        orig_dmrs_nxDG = nx.readwrite.json_graph.node_link_graph(filt_item['dmrs_nodelinkdict'])
        cleansed_dmrs_nxDG = drop_nodes(orig_dmrs_nxDG)
        if cleansed_dmrs_nxDG == None:
            del cleansed_sntid2item[snt_id]
            continue
        # reformat unknown predicate representation
        for node, node_prop in cleansed_dmrs_nxDG.nodes(data=True):
            pred = node_prop['instance']
            if pred.endswith("_u_unknown"):
                cleansed_dmrs_nxDG.nodes[node]['instance'] = change_unknown_pred_rep(pred)
        cleansed_sntid2item[snt_id]['dmrs_nodelinkdict'] = nx.readwrite.json_graph.node_link_data(cleansed_dmrs_nxDG)
        # adjust anchor to exclude preceding/trailing puntuations
        if not adjust_dmrs_anchor_boundaries(cleansed_dmrs_nxDG, cleansed_sntid2item[snt_id]['sentence']):
            print ("adj_anc err", snt_id)
            del cleansed_sntid2item[snt_id]
            continue
        cleansed_sntid2item[snt_id]['ancAdj_dmrs_nodelinkdict'] = nx.readwrite.json_graph.node_link_data(cleansed_dmrs_nxDG)

        ### cleanse derivation ###
        orig_deriv = nx.readwrite.json_graph.node_link_graph(filt_item['derivation_nodelinkdict'])
        # unify gigaword's derivation format
        if redwdOrGw == "gigaword":
            orig_deriv.add_node(None, entity = 'root_*')
            orig_deriv.add_edge(None, orig_deriv.graph['root'])
#         # adjust anchor, using the same function as above
#         adjust_deriv_anchor_boundaries(orig_deriv, cleansed_sntid2item[snt_id]['sentence'])
        cleansed_deriv = orig_deriv.copy()
        # propagate deriv anchoring and normalize terminal form
        deriv_root = list(cleansed_deriv.out_edges(nbunch = None))[0][0]
        anchor_from, anchor_to = propagate_anchors(cleansed_deriv, deriv_root)
        if anchor_from != "0" or anchor_to != str(len_snt):
            print (snt_id, cleansed_sntid2item[snt_id]['sentence'], ": propagation could be wrong")
            prop_err_cnt += 1
        cleansed_sntid2item[snt_id]['derivation_nodelinkdict'] = nx.readwrite.json_graph.node_link_data(cleansed_deriv)
        
        ### cleanse syntactic tree ###
        orig_syn_tree = nx.readwrite.json_graph.node_link_graph(filt_item['syn_tree_nodelinkdict'])
        cleansed_syn_tree = orig_syn_tree.copy()
        # check compatability of syntactic tree with derivation
        if len(orig_syn_tree.nodes) != len(orig_deriv) - 1:
            print ("Different deriv and syn_tree ", snt_id, len(orig_syn_tree.nodes), len(orig_deriv) - 1)
        # copy anchoring from derivation
        deriv_root2 = list(cleansed_deriv.out_edges(nbunch = [deriv_root]))[0][1]
        if redwdOrGw == 'gigaword':
            deriv_root2 = deriv_root
        if not syn_tree_anchors_from_deriv(cleansed_syn_tree, 0, cleansed_deriv, deriv_root2):
            print (snt_id, cleansed_sntid2item[snt_id]['sentence'], ": copy anchor could be wrong")
            copy_err_cnt += 1
        cleansed_sntid2item[snt_id]['syn_tree_nodelinkdict'] = nx.readwrite.json_graph.node_link_data(cleansed_syn_tree)
        
    sys.stderr.write('# probable err-ed anchor propagation item: {}\n'.format(prop_err_cnt))
    sys.stderr.write('# err-ed anchor copy item: {}\n'.format(copy_err_cnt))
#     print (len(cleansed_sntid2item))
    return cleansed_sntid2item

    
def main(redwoods_dir,
         gigaword_parsed_path,
         preprocessedData_dir_prefix,
         config_filepath,
         sampleOrAll = 'all',
         redwdOrGw = "redwoods",
         for_aceGen = False,
         pathForAce = None):
    
    
    
    sys.stderr.write("Loading config file ...\n")
    with open(config_filepath, "r", encoding='utf-8') as f:
        config = json.load(f)
    # pprint (config)
    global test_dir, alsoTest_dir, dev_dir, training_dir, bad_dir, virtual_dir
    test_dir = config['data_split']['test_profiles']
    alsoTest_dir = config['data_split']['also_test_profiles']
    dev_dir = config['data_split']['dev_profiles']
    training_dir = config['data_split']['training_profiles']
    bad_dir = config['data_split']['bad_profiles']
    virtual_dir = config['data_split']['virtual_dirs']
    ace_suffix = ""
    if for_aceGen: ace_suffix = "-ace"
    all_filt_data_filepath = os.path.join(preprocessedData_dir_prefix + "_" + redwdOrGw, "all",
                                          f'filtered_sntid2item{ace_suffix}.json')
    sample_filt_data_filepath = os.path.join(preprocessedData_dir_prefix + "_" + redwdOrGw, "all",
                                             f'sample_filtered_sntid2item{ace_suffix}.json')
    all_cleansed_data_filepath = os.path.join(preprocessedData_dir_prefix + "_" + redwdOrGw, "all",
                                              f'cleansed_sntid2item{ace_suffix}.json')
    sample_cleansed_data_filepath = os.path.join(preprocessedData_dir_prefix + "_" + redwdOrGw,  "all",
                                                 f'sample_cleansed_sntid2item{ace_suffix}.json')
    
    if sampleOrAll == 'all':
        filt_data_filepath = all_filt_data_filepath
        cleansed_data_filepath = all_cleansed_data_filepath
    elif sampleOrAll == 'sample':
        filt_data_filepath = sample_filt_data_filepath
        cleansed_data_filepath = sample_cleansed_data_filepath
    
    sys.stderr.write('Organizing data into dict...\n')
    filt_sntid2item = organize_to_dict(redwoods_dir, gigaword_parsed_path, sampleOrAll, redwdOrGw, for_aceGen)
    sys.stderr.write("Writing filtered data ...\n")
    os.makedirs(filt_data_filepath.rsplit("/", 1)[0], exist_ok=True)
    with open(filt_data_filepath, "w", encoding='utf-8') as f:
        json.dump(filt_sntid2item, f, indent=2)
    sys.stderr.write('Done! Written filtered data to file!\n')
    
    if not for_aceGen:
        sys.stderr.write('Cleansing dmrs, derivation and syntactic tree...\n')
        sys.stderr.write("Loading filtered data ...\n")
        with open(filt_data_filepath, "r", encoding='utf-8') as f:
            filt_sntid2item = json.load(f)
        sys.stderr.write("Loaded!\n")
        cleansed_sntid2item = cleanse_data(filt_sntid2item, redwdOrGw)
        sys.stderr.write("Writing cleansed data ...\n")
        os.makedirs(cleansed_data_filepath.rsplit("/", 1)[0], exist_ok=True)
        with open(cleansed_data_filepath, "w", encoding='utf-8') as f:
            json.dump(cleansed_sntid2item, f)#, indent=2)
        sys.stderr.write('Done! Written cleansed data to file!\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('redwoods_dir', default=None, help='path to redwoods directory')
    parser.add_argument('gigaword_parsed_path', default=None, help='path to gigaword parsed file path')
    parser.add_argument('preprocessedData_dir_prefix', help='path to preprocessed data directory')
    parser.add_argument('config_filepath', help='path to config file')
    parser.add_argument('sampleOrAll', default='all', help='try with sample data only or not')
    parser.add_argument('redwdOrGw', default='redwoods', help='preprocess redwoods or gigaword?')
    parser.add_argument('for_aceGen', default=False, help='prepare data for ACE generation')
    parser.add_argument('pathForAce', default=False, help='path to data prepared for ACE generation')
    args = parser.parse_args()
    main(args.redwoods_dir, args.gigaword_parsed_path, args.preprocessedData_dir_prefix,
         args.config_filepath, args.sampleOrAll, args.redwdOrGw, args.for_aceGen, arg.pathForAce)
