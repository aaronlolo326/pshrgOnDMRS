# print (snt_id)
# #util.write_figs_err(dmrs_nxDG, anno_deriv_nxDG, sentence)
# input()
# import delphin
# from delphin.codecs import simplemrs, dmrsjson
# from delphin import itsdb, util

import string
import traceback
import argparse
import pickle
import sys
import re
from pprint import pprint
from collections import defaultdict, Counter
import json
import os
import math
from functools import reduce
from ast import literal_eval as make_tuple
try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm
    
import networkx as nx
from networkx.algorithms.components import is_weakly_connected
from networkx.algorithms.boundary import edge_boundary
    
from src import dg_util
import src.util as util

max_subgraph_size = 6 # default as 6
lexicalized = False
node_typed = False
lex_min_freq = 100

rule2comma = Counter()
rule2noComma = Counter()
startRule2cnt = Counter()
ergRule2cnt = Counter()
dmrsSubgrCanon2bTag2cnt = defaultdict(Counter)
preTermCanon2bTagSurface2cnt = defaultdict(Counter)
preTerm2surface2cnt = defaultdict(Counter)
preTermAnno2surface2cnt = defaultdict(Counter)
preTermCanon2surface2cnt = defaultdict(Counter)
preTermCanonUsps2cnt = Counter() 
dgtrs2unaryRule2cnt = defaultdict(Counter)
# unaryRuleCcont2cnt = defaultdict(Counter)
dgtrs2binaryRule2cnt = defaultdict(Counter)
canon2cnt = Counter()
canon_usp2cnt = Counter()
# binaryRuleCcont2cnt = defaultdict(Counter)
# SHRG In the format of (concerned_nodes_repl, concerned_edges_repl): derivRule
# e.g. ((ccont_pred, erg_Lrule, erg_Rrule), (CL_edge_lbls, CR_edge_lbls, LR_edge_lbls)): erg_BinCcontRule
# e.g. ((erg_Urule), ()): erg_Crule
SHRG = defaultdict(Counter)
SHRG_coarse = defaultdict(Counter)
ccontCanon2intSubgrEdges2cnt = defaultdict(Counter)
interSubgrEdgeNo2cnt = Counter()
dgtrsEdgesTrg2SubtreeDgtrs = defaultdict(lambda: defaultdict(Counter))
eqAnc_semiCanons = set()
eqAnc_semiCanons_usp = set()
ccont_semiCanons = set()

def get_extractedFileName2data():
    global startRule2cnt, ergRule2cnt, dmrsSubgrCanon2bTag2cnt, preTermCanon2bTagSurface2cnt, preTerm2surface2cnt, preTermAnno2surface2cnt, preTermCanon2surface2cnt, preTermCanonUsps2cnt, dgtrs2unaryRule2cnt, dgtrs2binaryRule2cnt, SHRG, SHRG_coarse, ccontCanon2intSubgrEdges2cnt, dgtrsEdgesTrg2SubtreeDgtrs, canon2cnt, canon_usp2cnt, eqAnc_semiCanons, eqAnc_semiCanons_usp, ccont_semiCanons
    extracted_file2data = {
                    "startRule2cnt": startRule2cnt,
                    "ergRule2cnt": ergRule2cnt,
                     "dmrsSubgrCanon2bTag2cnt": dmrsSubgrCanon2bTag2cnt,
                     "preTermCanon2bTagSurface2cnt": preTermCanon2bTagSurface2cnt,
                     "preTerm2surface2cnt": preTerm2surface2cnt,
                     "preTermAnno2surface2cnt": preTermAnno2surface2cnt,
                     "preTermCanon2surface2cnt": preTermCanon2surface2cnt,
                     "preTermCanonUsps2cnt": preTermCanonUsps2cnt,
                     "dgtrs2unaryRule2cnt": dgtrs2unaryRule2cnt,
                     "dgtrs2binaryRule2cnt": dgtrs2binaryRule2cnt,
                     "SHRG": SHRG,
                     "SHRG_coarse": SHRG_coarse, 
                     "ccontCanon2intSubgrEdges2cnt": ccontCanon2intSubgrEdges2cnt,
                     "dgtrsEdgesTrg2SubtreeDgtrs": dgtrsEdgesTrg2SubtreeDgtrs,
                     "canon2cnt": canon2cnt,
                     "canon_usp2cnt": canon_usp2cnt,
                     "eqAnc_semiCanons": eqAnc_semiCanons,
                     "eqAnc_semiCanons_usp": eqAnc_semiCanons_usp,
                     "ccont_semiCanons": ccont_semiCanons}
    return extracted_file2data


def extract_equalanchor_subgraphs(training_sntid2item):
    
    all_subgraph_size2cnt = [0 for i in range(15)]
    con_subgraph_size2cnt = [0 for i in range(15)]
    large_con_subgraph2cnt = Counter()
    equalanchor_subgraphs_canon = Counter()
    equalanchor_subgraphs_semicanon = Counter()
    bridged_subgraphs_canon2neighbours = defaultdict(set)
    bridged_subgraphs_semicanon = set()
    
    for snt_id in tqdm(training_sntid2item):
        dmrs_nxDG_nodelink_dict = training_sntid2item[snt_id]["ancAdj_dmrs_nodelinkdict"]
        dmrs_nxDG = nx.readwrite.json_graph.node_link_graph(dmrs_nxDG_nodelink_dict)
        sentence = training_sntid2item[snt_id]["sentence"]

        anchors2nodes, anchors2preds = util.get_equalanchor_nodes(dmrs_nxDG, lexical_only = False)
        
        # for each anchor pair, check if the corresponding induced subgraph is weakly connected
        for anchors in anchors2preds:
            anchor_from,anchor_to  = anchors
            nodes_cnt = len(anchors2preds[anchors])
            # >=2 nodes form a subgraph
            if nodes_cnt >= 2 and nodes_cnt <= max_subgraph_size:
                all_subgraph_size2cnt[nodes_cnt] += 1
                node_ind_subgraph = dmrs_nxDG.subgraph(anchors2nodes[anchors])
                if is_weakly_connected(node_ind_subgraph):
                    con_subgraph_size2cnt[nodes_cnt] += 1
                    # node_ind_subgraph
                    ## exact canon
                    canonical_form, _, _ = util.get_canonical_form(node_ind_subgraph,dmrs_nxDG,extract_surfacemap = False)
                    semicanonical_form = util.get_semicanonical_form(anchors2preds[anchors])

                    equalanchor_subgraphs_canon[canonical_form] += 1
                    equalanchor_subgraphs_semicanon[semicanonical_form] += 1
                    

                else:
                    # to be done if disconnected (e.g. include bridging node)
                    # soln 1: include one node (record multiple choices of nodes if there are) as bridge if there is:
                    ## record those here
                    if not '-'  in sentence[int(anchor_from):int(anchor_to)]:
                        for node in dmrs_nxDG:
                            if node in anchors2nodes[anchors]:
                                continue
                            node_ind_subgraph = dmrs_nxDG.subgraph(anchors2nodes[anchors] + [node])
                            if is_weakly_connected(node_ind_subgraph):
                                canonical_form, sorted_nodes, _ = util.get_canonical_form(node_ind_subgraph,
                                                                                                     dmrs_nxDG,
                                                                                                     dummy_nodes=[node],
                                                                                                     extract_surfacemap = False)
                                node2nodeRepIdx = {node: idx for idx,node in enumerate(sorted_nodes)}
                                neighbours = set()
                                for src,targ,lbl in dmrs_nxDG.in_edges(node, data='label'):
                                    if not src in anchors2nodes[anchors]:
                                        continue
                                    neighbours.add(("src", node2nodeRepIdx[src], lbl))
                                for src,targ,lbl in dmrs_nxDG.out_edges(node, data='label'):
                                    if not targ in anchors2nodes[anchors]:
                                        continue
                                    neighbours.add(("targ", node2nodeRepIdx[targ], lbl))
                                bridged_subgraphs_canon2neighbours[canonical_form].add(frozenset(neighbours))

                                semicanonical_form = util.get_semicanonical_form(["dummy"] + anchors2preds[anchors])
                                bridged_subgraphs_semicanon.add(semicanonical_form)
                    # soln 2: ignore them here, and treat them during sync. rule extraction
            elif nodes_cnt > max_subgraph_size:
                node_ind_subgraph = dmrs_nxDG.subgraph(anchors2nodes[anchors])
                all_subgraph_size2cnt[nodes_cnt] += 1
                if is_weakly_connected(node_ind_subgraph):
                    con_subgraph_size2cnt[nodes_cnt] += 1
                    semicanonical_form = util.get_semicanonical_form(anchors2preds[anchors])
                    large_con_subgraph2cnt[semicanonical_form] += 1
                    
    return equalanchor_subgraphs_canon, equalanchor_subgraphs_semicanon, bridged_subgraphs_canon2neighbours, bridged_subgraphs_semicanon, all_subgraph_size2cnt, con_subgraph_size2cnt, large_con_subgraph2cnt

def align_dmrs_to_annoderiv(snt_id, dmrs_nxDG, anno_deriv_nxDG, sentence):
    # this function should serve to align only dmrs nodes/subgraphs to derivation's
    # preterminals/nearest binary rule in case of same dautgher anchors
    # syntactic construction has to be aligned during synchronous grammar extraction
    def _dfs_getAnchors2deriv_node(anchors2deriv_node, anno_deriv_nxDG, curr_node):
        curr_node_prop = anno_deriv_nxDG.nodes[curr_node]
        curr_anchor = (int(curr_node_prop['anchor_from']), int(curr_node_prop['anchor_to']))
        out_edges = list(anno_deriv_nxDG.out_edges(nbunch = [curr_node]))
        dgtrs_node = [targ for src, targ in out_edges]
        dgtrs_anchor = [(int(anno_deriv_nxDG.nodes[node]['anchor_from']),
                         int(anno_deriv_nxDG.nodes[node]['anchor_to'])) for node in dgtrs_node]
        # e.g. disconnected eqAnch deriv nodes, e.g. $300-a-share
        # tentatively, just pick the latest group of deriv nodes to align
        # TODO: ? 
        if curr_anchor in anchors2deriv_node:
            pass
        if len(out_edges) == 2:
            if curr_anchor in dgtrs_anchor and dgtrs_anchor[0] == dgtrs_anchor[1]:
                assert dgtrs_anchor[0][0] == dgtrs_anchor[1][0] or dgtrs_anchor[0][1] == dgtrs_anchor[1][1]
                anchors2deriv_node[curr_anchor] = curr_node
            else:
                assert dgtrs_anchor[0] != dgtrs_anchor[1]
                _dfs_getAnchors2deriv_node(anchors2deriv_node, anno_deriv_nxDG, dgtrs_node[0])
                _dfs_getAnchors2deriv_node(anchors2deriv_node, anno_deriv_nxDG, dgtrs_node[1])
                
        elif len(out_edges) == 1:
            assert curr_anchor == dgtrs_anchor[0]
            try: 
                if 'entity' in curr_node_prop\
                    and anno_deriv_nxDG.nodes[dgtrs_node[0]]['cat'] == '<terminal>':
                    anchors2deriv_node[curr_anchor] = curr_node
                else:
                    _dfs_getAnchors2deriv_node(anchors2deriv_node, anno_deriv_nxDG, dgtrs_node[0])
            except:
                for node, node_prop in anno_deriv_nxDG.nodes(data = True):
                    print (node, node_prop)
                #util.write_figs_err (None, anno_deriv_nxDG, "nocat")
                input()
        
    annoDerivPreTerm2dmrsNodeSubgr = defaultdict()
    dmrsNodeSubgr2preTerm = defaultdict()
    semEmtLexEnt = Counter()
    # build anchor2node for derivation
    anchors2deriv_node = defaultdict()
    
    _dfs_getAnchors2deriv_node(anchors2deriv_node, anno_deriv_nxDG, anno_deriv_nxDG.graph['root'])

    anchors2nodes, anchors2preds = util.get_equalanchor_nodes(dmrs_nxDG, lexical_only = False)
    # each node aligned once only
    node2isaligned = {node: False for node in dmrs_nxDG.nodes()}
    
    # for each anchor pair, check if the corresponding induced subgraph is weakly connected
    for anchors in anchors2preds:
        if anchors not in anchors2deriv_node:
            continue
        anchor_from, anchor_to = anchors
        nodes_cnt = len(anchors2preds[anchors])
        if any([node2isaligned[node] for node in anchors2nodes[anchors]]):
            print ("aligned:", node)
            input ()
            continue

        # >=2 nodes form a subgraph
        if nodes_cnt >= 2 and nodes_cnt <= max_subgraph_size:
            node_ind_subgraph = dmrs_nxDG.subgraph(anchors2nodes[anchors])
            if is_weakly_connected(node_ind_subgraph):
                for node in anchors2nodes[anchors]:
                    node2isaligned[node] = True
#                 canonical_form, sorted_nodes, _ = util.get_canonical_form(node_ind_subgraph,
#                                                                           dmrs_nxDG,
#                                                                           extract_surfacemap = False)
                annoDerivPreTerm2dmrsNodeSubgr[anchors2deriv_node[anchors]] = anchors2nodes[anchors]
#                 dmrsNodeSubgr2preTerm[canonical_form] = anno_deriv_nxDG.nodes[anchors2deriv_node[anchors]]['entity']
            else:
                # TODO (for disconnected subgraphs)
                # Just the same way as connected ones
                annoDerivPreTerm2dmrsNodeSubgr[anchors2deriv_node[anchors]] = anchors2nodes[anchors]
                for node in anchors2nodes[anchors]:
                    node2isaligned[node] = True
                
        # align a dmrs node to a deriv node
        elif nodes_cnt == 1:
            annoDerivPreTerm2dmrsNodeSubgr[anchors2deriv_node[anchors]] = anchors2nodes[anchors]
#             node_ind_subgraph = dmrs_nxDG.subgraph(anchors2nodes[anchors])
            for node in anchors2nodes[anchors]:
                node2isaligned[node] = True
#             canonical_form, sorted_nodes, _ = util.get_canonical_form(node_ind_subgraph,
#                                                                       dmrs_nxDG,
#                                                                       extract_surfacemap = False)
#             dmrsNodeSubgr2preTerm[canonical_form] = anno_deriv_nxDG.nodes[anchors2deriv_node[anchors]]['entity']
    # semEmtLexEnt
    for anchors in anchors2deriv_node:
#         out_edges = list(anno_deriv_nxDG.out_edges(nbunch = [anchors2deriv_node[anchors]]))
#         if len(out_edges) == 1:
#             dgtr = out_edges[0][1]
        if anchors not in anchors2nodes: # and anno_deriv_nxDG.nodes[dgtr]['cat'] == '<terminal>'
            semEmtLexEnt[anno_deriv_nxDG.nodes[anchors2deriv_node[anchors]]['entity']] += 1
    return annoDerivPreTerm2dmrsNodeSubgr, semEmtLexEnt # annoDerivPreTerm2dmrsNodeSubgr, dmrsNodeSubgr2preTerm

def extract_sync_grammar(snt_id, annoDerivPreTerm2dmrsNodeSubgr, dmrs_nxDG, dmrs_nxDG_ancAdj,
                         anno_deriv_nxDG, sentence, semEmtLexEnt_tmp):
        
    
    def _dfs_extract(annoDerivNode2dmrsNodeSubgr, annoDerivNode2dmrsNodeSubgr_repl, annoPretermNode2canon_usp, dmrs_node2isextracted, dmrs_edge2isextracted, dmrs_nxDG, dmrs_nxDG_repl, dmrs_nxDG_repl_orig, dmrs_nxDG_ancAdj, anno_deriv_nxDG_uni, curr_node, print_debug = False):
        global startRule2cnt, ergRule2cnt, dmrsSubgrCanon2bTag2cnt, preTermCanon2bTagSurface2cnt, preTerm2surface2cnt, preTermAnno2surface2cnt, preTermCanon2surface2cnt, preTermCanonUsps2cnt, dgtrs2unaryRule2cnt, dgtrs2binaryRule2cnt, SHRG, SHRG_coarse, ccontCanon2intSubgrEdges2cnt, dgtrsEdgesTrg2SubtreeDgtrs, interSubgrEdgeNo2cnt, canon2cnt, canon_usp2cnt, eqAnc_semiCanons, eqAnc_semiCanons_usp, rule2comma, rule2noComma

        for src, targ in anno_deriv_nxDG_uni.out_edges(nbunch = [curr_node]):
            if not anno_deriv_nxDG_uni.nodes[targ]['extracted']:
                _dfs_extract(annoDerivNode2dmrsNodeSubgr, annoDerivNode2dmrsNodeSubgr_repl, annoPretermNode2canon_usp, dmrs_node2isextracted, dmrs_edge2isextracted, dmrs_nxDG, dmrs_nxDG_repl, dmrs_nxDG_repl_orig, dmrs_nxDG_ancAdj, anno_deriv_nxDG_uni, targ)
        
        # propagate sematically empty lexical entry info to parent node
        # i.e. if any dgtr contains sem-emt related prop, propagate to curr_node
        # print (curr_node, "propagate error")
        # #util.write_figs_err(None, anno_deriv_nxDG_uni, None, None)
        # input()
        curr_derivRule = anno_deriv_nxDG_uni.nodes[curr_node]['entity']
        dgtrs = util.get_deriv_leftRight_dgtrs(anno_deriv_nxDG_uni, curr_node)
        dgtr_dmrs_nodes = [annoDerivNode2dmrsNodeSubgr.get(dgtr) or [] for dgtr in dgtrs]
        util.propagate_semEmt_in_deriv(anno_deriv_nxDG_uni, curr_node, dgtr_dmrs_nodes) 
        # comma propagation
        has_comma = False
        if len(dgtrs) == 2:
            has_comma = anno_deriv_nxDG_uni.nodes[dgtrs[0]]['comma']
        anno_deriv_nxDG_uni.nodes[curr_node]['comma'] = anno_deriv_nxDG_uni.nodes[dgtrs[-1]]['comma']
        if has_comma:
            rule2comma[curr_derivRule] += 1
        else:
            rule2noComma[curr_derivRule] += 1
            
        
        # extract if daugthers are extracted
        out_edges = list(anno_deriv_nxDG_uni.out_edges(nbunch = [curr_node]))
        if all([anno_deriv_nxDG_uni.nodes[targ]['extracted']
                for src, targ in out_edges]):
            mode = None
            # #daugther=2
            if len(out_edges) == 2:
                # first check if there's abstract pred associated to parent deriv node
                # e.g. compound, focus_d, ...
                # if there is, check if neighbouring nodes of dgtr's dmrs node/subgraph is with curr_deriv_anchor
#                 coarse_curr_derivRule = util.get_coarse_preTerm(curr_derivRule)
                curr_deriv_anchor = (int(anno_deriv_nxDG_uni.nodes[curr_node]['anchor_from']),
                                     int(anno_deriv_nxDG_uni.nodes[curr_node]['anchor_to']))
                dgtrs = util.get_deriv_leftRight_dgtrs(anno_deriv_nxDG_uni, curr_node)
                dgtrs_anchor = [(int(anno_deriv_nxDG_uni.nodes[node]['anchor_from']),
                                 int(anno_deriv_nxDG_uni.nodes[node]['anchor_to'])) for node in dgtrs]
                dgtrs_derivRule = (anno_deriv_nxDG_uni.nodes[dgtrs[0]]['entity'],
                                   anno_deriv_nxDG_uni.nodes[dgtrs[1]]['entity'])
#                 coarse_dgtrs_derivRule = dgtrs_derivRule
#                 coarse_dgtrs_derivRule = tuple(util.get_coarse_preTerm(dgtrs_derivRule[i])
#                                           if util.is_derivNode_preTerm(anno_deriv_nxDG_uni, dgtrs[i])
#                                           else dgtrs_derivRule[i]
#                                           for i in [0,1])

                left_dgtr_dmrs_nodes = annoDerivNode2dmrsNodeSubgr.get(dgtrs[0]) or []
                right_dgtr_dmrs_nodes = annoDerivNode2dmrsNodeSubgr.get(dgtrs[1]) or []
                left_dgtr_dmrs_nodes_repl = annoDerivNode2dmrsNodeSubgr_repl.get(dgtrs[0]) or []
                right_dgtr_dmrs_nodes_repl = annoDerivNode2dmrsNodeSubgr_repl.get(dgtrs[1]) or []
                left_dgtr_node_ind_subgraph = dmrs_nxDG.subgraph(left_dgtr_dmrs_nodes)
                right_dgtr_node_ind_subgraph = dmrs_nxDG.subgraph(right_dgtr_dmrs_nodes)
                # default
                ext_lRule, ext_rRule = None, None
                
                if curr_deriv_anchor in dmrs_anchors2nodes\
                    and all([not dmrs_node2isextracted[dmrs_node]
                             for dmrs_node in dmrs_anchors2nodes[curr_deriv_anchor]]):
                    # e.g. aj-hdn_norm_c of three-legged
                    if not left_dgtr_dmrs_nodes and not right_dgtr_dmrs_nodes:
                        
                        if curr_node in annoDerivNode2dmrsNodeSubgr:
                            mode = "BTag"
                            if print_debug:
                                print ("mode: {} on".format(mode), curr_node)
                            # print (util.get_surface_of_derivSubTree(anno_deriv_nxDG_uni, curr_node))
                            node_ind_subgraph = dmrs_nxDG.subgraph(annoDerivNode2dmrsNodeSubgr[curr_node])
                            node_ind_subgraph_repl\
                                = dmrs_nxDG_repl.subgraph(annoDerivNode2dmrsNodeSubgr_repl[curr_node])
                            for dmrs_node in annoDerivNode2dmrsNodeSubgr[curr_node]:
                                dmrs_node2isextracted[dmrs_node] = True
                            for edge in node_ind_subgraph.edges:
                                assert dmrs_edge2isextracted[edge] == False
                                dmrs_edge2isextracted[edge] = True
                            if is_weakly_connected(node_ind_subgraph):
                                canonical_form, sorted_nodes, _ = util.get_canonical_form(node_ind_subgraph_repl,
                                    dmrs_nxDG_repl_orig,
                                    extract_surfacemap = False, underspecLemma = False, underspecCarg = False)
                                uspCarg_canonical_form, sorted_nodes, _ = util.get_canonical_form(node_ind_subgraph_repl,
                                    dmrs_nxDG_repl_orig,
                                    extract_surfacemap = False, underspecLemma = False, underspecCarg = True, lexicalized = lexicalized)
                                usp_canonical_form, sorted_nodes_usp, _ = util.get_canonical_form(node_ind_subgraph_repl,
                                    dmrs_nxDG_repl_orig,
                                    extract_surfacemap = False, underspecLemma = True, underspecCarg = True, lexicalized = lexicalized)
                                semicanonical_form = util.get_semicanon_fromSubgr(node_ind_subgraph_repl,
                                                                                  underspecLemma = False)
                                interSubgrs_edges_lbls, interSubgrs_edges_key_orig = util.replace_subgraph(dmrs_nxDG_repl,
                                                                       dmrs_nxDG,
                                                                      annoDerivNode2dmrsNodeSubgr_repl[curr_node],
                                                                    [sorted_nodes],
                                                                    [True],
                                                                      curr_derivRule,
                                                                      curr_deriv_anchor,
                                                                      curr_node * 100 + 20000,
                                                                      mode = mode,
                                                                      node_typed = node_typed)
                                # rules collection
                                canon2cnt[str(canonical_form)] += 1
                                canon_usp2cnt[str(usp_canonical_form)] += 1
                                preTermCanon2bTagSurface2cnt[str(canonical_form)]\
                                    [sentence[curr_deriv_anchor[0]: curr_deriv_anchor[1]]] += 1
                                dmrsSubgrCanon2bTag2cnt[str(uspCarg_canonical_form)][curr_derivRule] += 1
#                                 print (canonical_form)
                                eqAnc_semiCanons.add(semicanonical_form)
#                                 dmrsSubgrCanonUsp2psdPreTerm2cnt[str(usp_canonical_form)]\
#                                     [util.get_coarse_preTerm(curr_derivRule)] += 1
                                # do not record the binary rule
                                # update deriv 2 dmrs_repl mapping and dmrs anchor to node mapping
                                # annoPretermNode2canon_usp[curr_node] = str(usp_canonical_form)
                                annoDerivNode2dmrsNodeSubgr_repl[curr_node] = [curr_node * 100 + 20000]
                                dmrs_anchors2nodes_repl[curr_deriv_anchor] = [curr_node * 100 + 20000]
                            # e.g. disconnected eqAnch dmrs nodes, e.g. $300-a-share offer
                            # [300] -1/EQ-> dollar (<-1/EQ- [a, share], <-2/NEQ- [compound])
                            # just proceed with bridgedSubgr2preTerm (TODO)
                            # dmrs replace also the bridging node (TODO)
                            else:
                                canonical_form, sorted_nodes, _ = util.get_canonical_form(node_ind_subgraph_repl,
                                    dmrs_nxDG_repl_orig,
                                    extract_surfacemap = False, underspecLemma = False, underspecCarg = False)
                                uspCarg_canonical_form, sorted_nodes, _ = util.get_canonical_form(node_ind_subgraph_repl,
                                    dmrs_nxDG_repl_orig,
                                    extract_surfacemap = False, underspecLemma = False, underspecCarg = True, lexicalized = lexicalized)
                                interSubgrs_edges_lbls, interSubgrs_edges_key_orig = util.replace_subgraph(dmrs_nxDG_repl,
                                                                       dmrs_nxDG,
                                                                      annoDerivNode2dmrsNodeSubgr_repl[curr_node],
                                                                    [sorted_nodes],
                                                                    [True],
                                                                      curr_derivRule,
                                                                      curr_deriv_anchor,
                                                                      curr_node * 100 + 20000,
                                                                      mode = mode, node_typed = node_typed)
                                # update deriv 2 dmrs_repl mapping
                                annoDerivNode2dmrsNodeSubgr_repl[curr_node] = [curr_node * 100 + 20000]
                                dmrs_anchors2nodes_repl[curr_deriv_anchor] = [curr_node * 100 + 20000]
#                                 print ("BTag disconnected")
#                                 print (canonical_form)
#                                 #util.write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, sentence)
#                                 input()
                        # multiple semantically empty words
                        else:
                            ext_lRule = dgtrs_derivRule[0]
                            ext_rRule = dgtrs_derivRule[1]
                        
                    # predicate-bearing binary rule
                    else:
                        mode = "BP"
                        if print_debug:
                            print ("mode: {} on".format(mode), curr_node)
                        curr_node_ind_subgraph = dmrs_nxDG.subgraph(dmrs_anchors2nodes[curr_deriv_anchor])
                        curr_node_ind_subgraph_repl = dmrs_nxDG_repl.subgraph(dmrs_anchors2nodes_repl[curr_deriv_anchor])
                        annoDerivNode2dmrsNodeSubgr[curr_node] = list(set(left_dgtr_dmrs_nodes\
                            + right_dgtr_dmrs_nodes + dmrs_anchors2nodes[curr_deriv_anchor]))
                        annoDerivNode2dmrsNodeSubgr_repl[curr_node] = list(set(left_dgtr_dmrs_nodes_repl\
                            + right_dgtr_dmrs_nodes_repl + dmrs_anchors2nodes_repl[curr_deriv_anchor]))
                        interSubgrLR_edges_key, interSubgrLR_edges_lbl = util.get_interSubgr_edges(dmrs_nxDG,
                            left_dgtr_dmrs_nodes,
                            right_dgtr_dmrs_nodes, directed = False)
                        interSubgrCL_edges_key, interSubgrCL_edges_lbl = util.get_interSubgr_edges(dmrs_nxDG,
                            dmrs_anchors2nodes[curr_deriv_anchor],
                            left_dgtr_dmrs_nodes, directed = False)
                        interSubgrCR_edges_key, interSubgrCR_edges_lbl = util.get_interSubgr_edges(dmrs_nxDG,
                            dmrs_anchors2nodes[curr_deriv_anchor],
                            right_dgtr_dmrs_nodes, directed = False)
                        interSubgrEdgeNo2cnt[len(interSubgrLR_edges_key)] += 1
                        interSubgrEdgeNo2cnt[len(interSubgrCL_edges_key)] += 1
                        interSubgrEdgeNo2cnt[len(interSubgrCR_edges_key)] += 1
                        for dmrs_node in dmrs_anchors2nodes[curr_deriv_anchor]:
                            dmrs_node2isextracted[dmrs_node] = True
                        for edge in curr_node_ind_subgraph.edges:
                            assert dmrs_edge2isextracted[edge] == False
                            dmrs_edge2isextracted[edge] = True
                        for edge in interSubgrLR_edges_key:
                            assert dmrs_edge2isextracted[edge] == False
                            dmrs_edge2isextracted[edge] = True
                        for edge in interSubgrCL_edges_key:
                            assert dmrs_edge2isextracted[edge] == False
                            dmrs_edge2isextracted[edge] = True
                        for edge in interSubgrCR_edges_key:
                            assert dmrs_edge2isextracted[edge] == False
                            dmrs_edge2isextracted[edge] = True
                        # e.g. I [have eaten]
                        if not left_dgtr_dmrs_nodes or not right_dgtr_dmrs_nodes:
                            # In fact, we should not replace subgraph fro BP coz it should be replaced altogether
                            # with a subtree when the sem-emt is recorded, but for simplicity we still replace it;
                            # The consequence is that upon testing, the subtree cannot be used because we prematurely
                            # interacted with the introduced predicate;
                            # The interaction between the introduced pred with the only dgtr subgraph
                            # is already generalized below
                            pass
                        # extract abstract pred's interaction with the two subgraphs
                        # pay attention to ineteraction in examples like
                        # "The compound[old and garden] dog barked."
                        # if weakly connected, do replacement of dmrs subgraphs
                        old_dmrs_nxDG_repl = dmrs_nxDG_repl.copy()
                        if is_weakly_connected(curr_node_ind_subgraph_repl):
                            # record the left and right deriv tule of dmrs_nodes_repl
                            lRule = None
                            rRule = None
                            if left_dgtr_dmrs_nodes_repl:
                                lRule = dmrs_nxDG_repl.nodes[left_dgtr_dmrs_nodes_repl[0]]['derivRule_usp']
#                                 lRule = util.get_coarse_preTerm(lRule)
                            if right_dgtr_dmrs_nodes_repl:
                                rRule = dmrs_nxDG_repl.nodes[right_dgtr_dmrs_nodes_repl[0]]['derivRule_usp']
#                                 rRule = util.get_coarse_preTerm(rRule)
                            canonical_form, sorted_nodes, _ = util.get_canonical_form(curr_node_ind_subgraph_repl,
                                dmrs_nxDG_repl_orig, extract_surfacemap = False)
                            usp_canonical_form, sorted_nodes_usp, _ = util.get_canonical_form(curr_node_ind_subgraph_repl,
                                    dmrs_nxDG_repl_orig,
                                    extract_surfacemap = False, underspecLemma = True, underspecCarg = True)
                            semicanonical_form = util.get_semicanon_fromSubgr(curr_node_ind_subgraph_repl)
#                             util.write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, sentence)
                            interSubgrs_edges_lbls, interSubgrs_edges_key_orig = util.replace_subgraph(dmrs_nxDG_repl,
                                                                           dmrs_nxDG,
                                                            annoDerivNode2dmrsNodeSubgr_repl[curr_node],
                                    [sorted_nodes, left_dgtr_dmrs_nodes_repl, right_dgtr_dmrs_nodes_repl],
                                                            [True, False, False],
                                                                  curr_derivRule,
                                                                  curr_deriv_anchor, 
                                                             curr_node * 100 + 20000,
                                                          mode = mode, node_typed = node_typed)
#                             util.write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, sentence)
#                             print (interSubgrs_edges_lbls)
                            # update deriv 2 dmrs_repl mapping
                            annoDerivNode2dmrsNodeSubgr_repl[curr_node] = [curr_node * 100 + 20000]
                            dmrs_anchors2nodes_repl[curr_deriv_anchor] = [curr_node * 100 + 20000]
                            # record interaction of introduced dmrs nodes with left and right
                            # dmrs subgraphs respectively
                            if interSubgrCL_edges_key or interSubgrCR_edges_key:
                                # rules collection
                                SHRG[str(((canonical_form, lRule,  rRule), interSubgrs_edges_lbls))][curr_derivRule] += 1
                                SHRG_coarse[str(((canonical_form, lRule,  rRule), interSubgrs_edges_lbls))][curr_derivRule] += 1
                                ccontCanon2intSubgrEdges2cnt[str(canonical_form)][str(interSubgrs_edges_lbls)] += 1
                                ccont_semiCanons.add(semicanonical_form)
                                canon2cnt[str(canonical_form)] += 1
                                canon_usp2cnt[str(usp_canonical_form)] += 1
                                ext_lRule = lRule or dgtrs_derivRule[0]
                                ext_rRule = rRule or dgtrs_derivRule[1]
                                # handle semEmt for e.g. phrasal verb, copula
                                # extract subtree if both dgtr is anchored and either has trg_in_subtree and
                                # semEmt_in_subtree matched
                                # subtree starts from curr_node up to those descendents that are anchored 
                                if left_dgtr_dmrs_nodes and right_dgtr_dmrs_nodes:
                                    for semEmt_type in ['copula', 'prtcl', 'compl', 'by', 'predSemEmt']:
                                        if semEmt_type == 'predSemEmt':
                                            predsAndSemEmts = util.get_predTrgSemEmt(dmrs_nxDG, interSubgrs_edges_key_orig,
                                                                                     anno_deriv_nxDG_uni, curr_node)
                                            for pred, semEmt, interSubgrs_edge_lbl, edgeSet_idx in predsAndSemEmts:
                                                subtree, left_cfg_dgtr_ent, right_cfg_dgtr_ent,\
                                                    left_node, right_node = util.extract_semEmts_subtree(anno_deriv_nxDG_uni,
                                                                                         annoDerivNode2dmrsNodeSubgr,
                                                                                              annoPretermNode2canon_usp,
                                                                                         curr_node, (semEmt,),
                                                                                            (pred, None),
                                                                                             dmrs_nxDG_repl, dmrs_nxDG,
                                                                                              semEmtType = semEmt_type)
                                                if subtree:
                                                    left_cfg_dgtr_ent = annoPretermNode2canon_usp.get(left_node) or left_cfg_dgtr_ent
                                                    right_cfg_dgtr_ent = annoPretermNode2canon_usp.get(right_node) or right_cfg_dgtr_ent
                                                    semEmt_key = (canonical_form, left_cfg_dgtr_ent, right_cfg_dgtr_ent)
                                                    subtree_dict = nx.readwrite.json_graph.tree_data(subtree, root = curr_node)
                                                    dgtrsEdgesTrg2SubtreeDgtrs[semEmt_type]\
                                                        [str(((semEmt_key, (pred, interSubgrs_edge_lbl, edgeSet_idx)),
                                                              interSubgrs_edges_lbls))]\
                                                        [str((subtree_dict, left_node, right_node))] += 1

                                        else:
                                            matched_semEmts, targ_semEmt_trgL, targ_semEmt_trgR = util.is_semEmt_extractable(
                                                anno_deriv_nxDG_uni, curr_node, semEmtType = semEmt_type)
                                            if matched_semEmts:
        #                                         print (matched_semEmts, f'is {semEmt_type}-matched', (targ_semEmt_trgL, targ_semEmt_trgR), "BP")
                                                subtree, left_cfg_dgtr_ent, right_cfg_dgtr_ent,\
                                                    left_node, right_node = util.extract_semEmts_subtree(anno_deriv_nxDG_uni,
                                                                                         annoDerivNode2dmrsNodeSubgr,
                                                                                              annoPretermNode2canon_usp,
                                                                                         curr_node, matched_semEmts,
                                                                                            (targ_semEmt_trgL, targ_semEmt_trgR),
                                                                                             dmrs_nxDG_repl, dmrs_nxDG,
                                                                                              semEmtType = semEmt_type)
                                                if subtree:
                                                    left_cfg_dgtr_ent = annoPretermNode2canon_usp.get(left_node) or left_cfg_dgtr_ent
                                                    right_cfg_dgtr_ent = annoPretermNode2canon_usp.get(right_node) or right_cfg_dgtr_ent
                                                    semEmt_key = ((canonical_form, None), (left_cfg_dgtr_ent, targ_semEmt_trgL),
                                                                  (right_cfg_dgtr_ent, targ_semEmt_trgR))
                                                    subtree_dict = nx.readwrite.json_graph.tree_data(subtree, root = curr_node)
                                                    dgtrsEdgesTrg2SubtreeDgtrs[semEmt_type][str((semEmt_key, interSubgrs_edges_lbls))]\
                                                        [str((subtree_dict, left_node, right_node))] += 1

                            else:
                                print (curr_node, "no CL and CR?")
                                ext_lRule = lRule or dgtrs_derivRule[0]
                                ext_rRule = rRule or dgtrs_derivRule[1]

                        # ignore if the introduced preds are disconnected
                        # when would this happend?
                        # e.g. two unary rule each introduce a pred, equally anchored
                        else:
                            # record the left and right deriv rule of dmrs_nodes_repl
                            lRule = None
                            rRule = None
                            if left_dgtr_dmrs_nodes_repl:
                                lRule = dmrs_nxDG_repl.nodes[left_dgtr_dmrs_nodes_repl[0]]['derivRule_usp']
                            if right_dgtr_dmrs_nodes_repl:
                                rRule = dmrs_nxDG_repl.nodes[right_dgtr_dmrs_nodes_repl[0]]['derivRule_usp']
#                                 rRule = util.get_coarse_preTerm(rRule)
                            canonical_form, sorted_nodes, _ = util.get_canonical_form(curr_node_ind_subgraph_repl,
                                dmrs_nxDG_repl_orig,
                                extract_surfacemap = False)
                            interSubgrs_edges_lbls, interSubgrs_edges_key_orig = util.replace_subgraph(dmrs_nxDG_repl,
                                                                           dmrs_nxDG,
                                                            annoDerivNode2dmrsNodeSubgr_repl[curr_node],
                                    [sorted_nodes, left_dgtr_dmrs_nodes_repl, right_dgtr_dmrs_nodes_repl],
                                                            [True, False, False],
                                                                  curr_derivRule,
                                                                  curr_deriv_anchor,
                                                             curr_node * 100 + 20000,
                                                          mode = mode, node_typed = node_typed)
                            # update deriv 2 dmrs_repl mapping
                            annoDerivNode2dmrsNodeSubgr_repl[curr_node] = [curr_node * 100 + 20000]
                            dmrs_anchors2nodes_repl[curr_deriv_anchor] = [curr_node * 100 + 20000]
                            ext_lRule = lRule or dgtrs_derivRule[0]
                            ext_rRule = rRule or dgtrs_derivRule[1]
                                      
                else:
                    mode = "B"
                    if print_debug:
                        print ("mode: {} on".format(mode), curr_node)
                        
                    # check inter-subgraph edges
                    interSubgrLR_edges_key, interSubgrLR_edges_lbl = util.get_interSubgr_edges(dmrs_nxDG,
                        left_dgtr_dmrs_nodes,
                        right_dgtr_dmrs_nodes, directed = False)
                    for edge in interSubgrLR_edges_key:
                        assert dmrs_edge2isextracted[edge] == False
                        dmrs_edge2isextracted[edge] = True
                    annoDerivNode2dmrsNodeSubgr[curr_node] = list(set(left_dgtr_dmrs_nodes\
                        + right_dgtr_dmrs_nodes))
                    annoDerivNode2dmrsNodeSubgr_repl[curr_node] = list(set(left_dgtr_dmrs_nodes_repl\
                        + right_dgtr_dmrs_nodes_repl))
                    # record the left and right deriv tule of dmrs_nodes_repl
                    lRule = None
                    rRule = None
                    if left_dgtr_dmrs_nodes_repl:
                        lRule = dmrs_nxDG_repl.nodes[left_dgtr_dmrs_nodes_repl[0]]['derivRule_usp']
                        lRule_src = dmrs_nxDG_repl.nodes[left_dgtr_dmrs_nodes_repl[0]]['derivRule']
#                         lRule = util.get_coarse_preTerm(lRule)
                    if right_dgtr_dmrs_nodes_repl:
                        rRule = dmrs_nxDG_repl.nodes[right_dgtr_dmrs_nodes_repl[0]]['derivRule_usp']
                        rRule_src = dmrs_nxDG_repl.nodes[right_dgtr_dmrs_nodes_repl[0]]['derivRule']
#                         rRule = util.get_coarse_preTerm(rRule)
                    ext_lRule = lRule or dgtrs_derivRule[0]
                    ext_rRule = rRule or dgtrs_derivRule[1]
#                     print (ext_lRule, ext_rRule)
#                     util.write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, sentence, ext_lRule+ext_rRule)
                    # replace subgraph occurs here
                    interSubgrs_edges_lbls, interSubgrs_edges_key_orig = util.replace_subgraph(dmrs_nxDG_repl,
                                               dmrs_nxDG,
                                          annoDerivNode2dmrsNodeSubgr_repl[curr_node],
                                    [left_dgtr_dmrs_nodes_repl, right_dgtr_dmrs_nodes_repl],
                                     [False, False],
                                          curr_derivRule,
                                          curr_deriv_anchor,
                                          curr_node * 100 + 20000,
                                          mode = mode, node_typed = node_typed) 
                    # update deriv 2 dmrs_repl mapping
                    annoDerivNode2dmrsNodeSubgr_repl[curr_node] = [curr_node * 100 + 20000]
                    dmrs_anchors2nodes_repl[curr_deriv_anchor] = [curr_node * 100 + 20000]
                    
                    # extract rule only if there's interaction betwn L R dmrs subgraphs
                    if interSubgrLR_edges_key:
                        # fallback
                        if lRule == None or rRule == None:
                            print ("1!")
                            print (interSubgrs_edges_lbls)
                            print (left_dgtr_dmrs_nodes_repl)
                            print (right_dgtr_dmrs_nodes_repl)
                            print (lRule_src)
                            print (rRule_src)
                            print (lRule, rRule)
                            input()
                            
                        SHRG_coarse[str(((lRule,  rRule), interSubgrs_edges_lbls))][curr_derivRule] += 1 
                        SHRG[str(((lRule,  rRule), interSubgrs_edges_lbls))][curr_derivRule] += 1
                        exactDiff = False
                        # record the left and right deriv tule of dmrs_nodes_repl
                        if left_dgtr_dmrs_nodes_repl:
                            if interSubgrs_edges_lbls[0] and not interSubgrs_edges_lbls[1]:
                                if lRule != lRule_src:
#                                     print (lRule, lRule_src)
#                                     print (interSubgrs_edges_lbls)
                                    lRule = lRule_src
                                    exactDiff = True
                        if right_dgtr_dmrs_nodes_repl:
                            if interSubgrs_edges_lbls[1] and not interSubgrs_edges_lbls[0]:
                                if rRule != rRule_src:
                                    rRule = rRule_src
#                                     print (rRule, rRule_src)
#                                     print (interSubgrs_edges_lbls)
                                    exactDiff = True
#                         if "[usp]" in str(lRule) and "[usp]" in str(rRule):
#                             print (interSubgrs_edges_lbls)
#                             print (left_dgtr_dmrs_nodes_repl)
#                             print (right_dgtr_dmrs_nodes_repl)
#                             print (lRule_src)
#                             print (rRule_src)
#                             input()
#                         ext_lRule = lRule or dgtrs_derivRule[0]
#                         ext_rRule = rRule or dgtrs_derivRule[1]
                        # rules collection
                        if lRule == None or rRule == None:
                            print ("2!")
                            print (interSubgrs_edges_lbls)
                            print (left_dgtr_dmrs_nodes_repl)
                            print (right_dgtr_dmrs_nodes_repl)
                            print (lRule_src)
                            print (rRule_src)
                            print (lRule, rRule)
                            input()
                        if exactDiff:
                            SHRG[str(((lRule,  rRule), interSubgrs_edges_lbls))][curr_derivRule] += 1
#                             print ()
                    
                        interSubgrEdgeNo2cnt[len(interSubgrLR_edges_key)] += 1
                        # extract subtree if both dgtr is anchored and either has trg_in_subtree and
                        # semEmt_in_subtree matched
                        # subtree starts from curr_node up to those descendents that are anchored 
                        if left_dgtr_dmrs_nodes and right_dgtr_dmrs_nodes:
                            for semEmt_type in ['copula', 'prtcl', 'compl', 'by', 'predSemEmt']:
                                if semEmt_type == 'predSemEmt':
                                    predsAndSemEmts = util.get_predTrgSemEmt(dmrs_nxDG, interSubgrs_edges_key_orig,
                                                                             anno_deriv_nxDG_uni, curr_node)
                                    for pred, semEmt, interSubgrs_edge_lbl, edgeSet_idx in predsAndSemEmts:
                                        subtree, left_cfg_dgtr_ent, right_cfg_dgtr_ent,\
                                            left_node, right_node = util.extract_semEmts_subtree(anno_deriv_nxDG_uni,
                                                                                 annoDerivNode2dmrsNodeSubgr,
                                                                                      annoPretermNode2canon_usp,
                                                                                 curr_node, (semEmt,),
                                                                                    (pred, None),
                                                                                     dmrs_nxDG_repl, dmrs_nxDG,
                                                                                      semEmtType = semEmt_type)
                                        if subtree:
                                            left_cfg_dgtr_ent = annoPretermNode2canon_usp.get(left_node) or left_cfg_dgtr_ent
                                            right_cfg_dgtr_ent = annoPretermNode2canon_usp.get(right_node) or right_cfg_dgtr_ent
                                            semEmt_key = (left_cfg_dgtr_ent, right_cfg_dgtr_ent)
                                            subtree_dict = nx.readwrite.json_graph.tree_data(subtree, root = curr_node)
                                            dgtrsEdgesTrg2SubtreeDgtrs[semEmt_type]\
                                                [str(((semEmt_key, (pred, interSubgrs_edge_lbl, edgeSet_idx)), interSubgrs_edges_lbls))]\
                                                [str((subtree_dict, left_node, right_node))] += 1

                                else:
                                    matched_semEmts, targ_semEmt_trgL, targ_semEmt_trgR = util.is_semEmt_extractable(
                                        anno_deriv_nxDG_uni, curr_node, semEmtType = semEmt_type)
                                    
                                    if matched_semEmts:
#                                         print (matched_semEmts, f'is {semEmt_type}-matched', (targ_semEmt_trgL, targ_semEmt_trgR), "BP")
                                        subtree, left_cfg_dgtr_ent, right_cfg_dgtr_ent,\
                                            left_node, right_node = util.extract_semEmts_subtree(anno_deriv_nxDG_uni,
                                                                                 annoDerivNode2dmrsNodeSubgr,
                                                                                      annoPretermNode2canon_usp,
                                                                                 curr_node, matched_semEmts,
                                                                                    (targ_semEmt_trgL, targ_semEmt_trgR),
                                                                                     dmrs_nxDG_repl, dmrs_nxDG,
                                                                                      semEmtType = semEmt_type)
                                        if subtree:
                                            left_cfg_dgtr_ent = annoPretermNode2canon_usp.get(left_node) or left_cfg_dgtr_ent
                                            right_cfg_dgtr_ent = annoPretermNode2canon_usp.get(right_node) or right_cfg_dgtr_ent
                                            semEmt_key = ((left_cfg_dgtr_ent, targ_semEmt_trgL),
                                                          (right_cfg_dgtr_ent, targ_semEmt_trgR))
                                            subtree_dict = nx.readwrite.json_graph.tree_data(subtree, root = curr_node)
                                            dgtrsEdgesTrg2SubtreeDgtrs[semEmt_type][str((semEmt_key, interSubgrs_edges_lbls))]\
                                                [str((subtree_dict, left_node, right_node))] += 1
                                   
                    else:
                        # e.g. disconnected dmrs subgraphs/nodes               
                        # e.g. (part)--([of]--[which])* (not left_dgtr_dmrs_nodes and not right_dgtr_dmrs_nodes)
                        # e.g. I [have eaten] (not left_dgtr_dmrs_nodes or not right_dgtr_dmrs_nodes:)
                        # e.g. have, will, particles like run [into], etc
                        pass
                    
                if ext_lRule and ext_rRule:
                    dgtrs2binaryRule2cnt[str((ext_lRule, ext_rRule))][curr_derivRule] += 1
#                     print (ext_lRule, ext_rRule)
                    ergRule2cnt[curr_derivRule] += 1
            # #daugther=1
            elif len(out_edges) == 1:
                dgtr = out_edges[0][1]
                curr_derivRule = anno_deriv_nxDG_uni.nodes[curr_node]['entity']
#                 coarse_curr_derivRule = util.get_coarse_preTerm(curr_derivRule)
                curr_deriv_anchor = (int(anno_deriv_nxDG_uni.nodes[curr_node]['anchor_from']),
                                     int(anno_deriv_nxDG_uni.nodes[curr_node]['anchor_to']))
                dgtr_deriv_anchor = (int(anno_deriv_nxDG_uni.nodes[dgtr]['anchor_from']),
                                     int(anno_deriv_nxDG_uni.nodes[dgtr]['anchor_to']))
                
                dgtr_derivRule = (anno_deriv_nxDG_uni.nodes[dgtr].get('entity'))\
                    or None
#                 coarse_dgtr_derivRule = dgtr_derivRule
#                 if dgtr_derivRule and util.is_derivNode_preTerm(anno_deriv_nxDG_uni, dgtr):
#                     coarse_dgtr_derivRule = (util.get_coarse_preTerm(dgtr_derivRule))
#                     print (dgtr_derivRule, coarse_dgtr_derivRule)
                assert curr_deriv_anchor == dgtr_deriv_anchor
                dgtr_dmrs_nodes = annoDerivNode2dmrsNodeSubgr.get(dgtr) or []
                dgtr_dmrs_nodes_repl = annoDerivNode2dmrsNodeSubgr_repl.get(dgtr) or []
                # rule frequency collection
#                 if dgtr_derivRule:
                    # print ((curr_derivRule, dgtr_derivRule))
                uRule = None
                # case of semantically empty lexeme, or "three-" and "legged" in three-legged dog
                if anno_deriv_nxDG_uni.nodes[dgtr]['cat'] == '<terminal>':
                    preTerm2surface2cnt[curr_derivRule.split("/")[0]]\
                        [anno_deriv_nxDG_uni.nodes[dgtr]['form']] += 1
                    preTermAnno2surface2cnt[curr_derivRule][anno_deriv_nxDG_uni.nodes[dgtr]['form']] += 1
                # either dgtr is terminal or dgtr dmrs is empty
                if anno_deriv_nxDG_uni.nodes[dgtr]['cat'] == '<terminal>'\
                    or not dgtr_dmrs_nodes:
                    if curr_node not in annoDerivNode2dmrsNodeSubgr:
                        if anno_deriv_nxDG_uni.nodes[dgtr]['cat'] == '<terminal>':
                            semEmpLex_cand = anno_deriv_nxDG_uni.nodes[curr_node]['entity']
                            if util.is_copula(semEmpLex_cand):
                                anno_deriv_nxDG_uni.nodes[curr_node]['copula'].append(semEmpLex_cand)
                            if util.is_prtcl(semEmpLex_cand):
                                anno_deriv_nxDG_uni.nodes[curr_node]['prtcl'].append(semEmpLex_cand)
                            if util.is_compl(semEmpLex_cand):
                                anno_deriv_nxDG_uni.nodes[curr_node]['compl'].append(semEmpLex_cand)
                            if util.is_predSemEmt(semEmpLex_cand):
                                anno_deriv_nxDG_uni.nodes[curr_node]['predSemEmt'].append(semEmpLex_cand)
                            if util.is_by(semEmpLex_cand):
                                anno_deriv_nxDG_uni.nodes[curr_node]['by'].append(semEmpLex_cand)
                            
                        else:
                            uRule = dgtr_derivRule
                            
                    elif all([not dmrs_node2isextracted[dmrs_node]
                              for dmrs_node in annoDerivNode2dmrsNodeSubgr[curr_node]]):
                        mode = "Utag"
                        if print_debug:
                            print ("mode: {} on".format(mode), curr_node)
                        # print ("preterm extract now:", curr_node)
                        node_ind_subgraph = dmrs_nxDG.subgraph(annoDerivNode2dmrsNodeSubgr[curr_node])
                        node_ind_subgraph_repl = dmrs_nxDG_repl.subgraph(annoDerivNode2dmrsNodeSubgr_repl[curr_node])
                        for dmrs_node in annoDerivNode2dmrsNodeSubgr[curr_node]:
                            dmrs_node2isextracted[dmrs_node] = True
                        for edge in node_ind_subgraph.edges:
                            assert dmrs_edge2isextracted[edge] == False
                            dmrs_edge2isextracted[edge] = True
                        if is_weakly_connected(node_ind_subgraph_repl):
                            # for surface
                            surface_canonical_form, sorted_nodes, _ = util.get_canonical_form(node_ind_subgraph_repl,
                                dmrs_nxDG_repl_orig,
                                extract_surfacemap = False, underspecLemma = False, underspecCarg = False, forSurfGen = True)
                            # for head
                            uspCarg_canonical_form, sorted_nodes_uspCarg, _ = util.get_canonical_form(node_ind_subgraph_repl,
                                dmrs_nxDG_repl_orig,
                                extract_surfacemap = False, underspecCarg = True, lexicalized = lexicalized)
                            # for tail
                            usp_canonical_form, sorted_nodes_usp, _ = util.get_canonical_form(node_ind_subgraph_repl,
                                dmrs_nxDG_repl_orig,
                                extract_surfacemap = False, underspecLemma = True, underspecCarg = True, lexicalized = lexicalized)
                            semicanonical_form = util.get_semicanon_fromSubgr(node_ind_subgraph_repl,
                                                                              underspecLemma = False)
                            semicanonical_form_usp = util.get_semicanon_fromSubgr(node_ind_subgraph_repl,
                                                                              underspecLemma = True)
                            ruleToReplace = uspCarg_canonical_form
#                             if lexicalized:
#                                 if len(sorted_nodes) == 1:
#                                     ruleToReplace = dmrs_nxDG.nodes[annoDerivNode2dmrsNodeSubgr[curr_node]]['instance']
                            interSubgrs_edges_lbls, interSubgrs_edges_key_orig = util.replace_subgraph(dmrs_nxDG_repl,
                                                                       dmrs_nxDG,
                                                                       annoDerivNode2dmrsNodeSubgr_repl[curr_node],
                                                                  [sorted_nodes_uspCarg],
                                                                    [True],
                                                                  ruleToReplace,
                                                                  curr_deriv_anchor,
                                                                  curr_node * 100 + 20000,
                                                              mode = mode, derivRule_usp = usp_canonical_form, node_typed = node_typed)
                            # handle semEmt for e.g. phrasal verb, copula
                            # print ("look for trg")
                            for node in dmrs_anchors2nodes[curr_deriv_anchor]:
                                # prtcl trg
                                prtcl = util.get_pred_prtcl(dmrs_nxDG.nodes[node]['instance'])
                                if prtcl:
                                    prtcl_trg = util.get_prtcl_trg(dmrs_nxDG, node, prtcl)
                                    anno_deriv_nxDG_uni.nodes[curr_node]['prtcl_trg'] = (prtcl_trg,
                                                                                         dmrs_nxDG.nodes[node]['instance'],
                                                                                         prtcl)
#                                     print ("prtcl trg: ", dmrs_nxDG.nodes[node], canonical_form, mode)
                                # copula trg
                                copula_trg, tense = util.get_copula_trg(dmrs_nxDG, node)
                                if copula_trg:
                                    if (not anno_deriv_nxDG_uni.nodes[curr_node]['copula_trg'] or tense != 'UNTENSED'):
                                        anno_deriv_nxDG_uni.nodes[curr_node]['copula_trg'] = copula_trg
    #                                     print ("copula trg: ", dmrs_nxDG.nodes[node], copula_trg, mode)
                                # compl trg
                                compl_trg = util.get_compl_trg(dmrs_nxDG, node)
                                if compl_trg:
                                    anno_deriv_nxDG_uni.nodes[curr_node]['compl_trg'] = compl_trg
                                
                             # rules collection
                            ancAdj_surface = \
                                util.get_surface_ofNodes(
                                    [dmrs_nxDG_ancAdj.nodes[node] for node in annoDerivNode2dmrsNodeSubgr[curr_node]],
                                    sentence
                                )
                            canon2cnt[str(surface_canonical_form)] += 1
                            canon_usp2cnt[str(usp_canonical_form)] += 1
#                             surfaceCanon = util.node_toString(annoDerivNode2dmrsNodeSubgr[curr_node][0],
#                                                                dmrs_nxDG.nodes[annoDerivNode2dmrsNodeSubgr[curr_node][0]],
#                                                               dmrs_nxDG, add_posSpecInfo = True, add_surfaceInfo = True)
                            preTermCanon2surface2cnt[str(surface_canonical_form)][ancAdj_surface] += 1
                            eqAnc_semiCanons.add(semicanonical_form)
                            eqAnc_semiCanons_usp.add(semicanonical_form_usp)
                            if not lexicalized:
                                preTermCanonUsps2cnt[str(usp_canonical_form)] += 1
                                annoPretermNode2canon_usp[curr_node] = usp_canonical_form
                            else:
                                if len(sorted_nodes) >= 2:
                                    preTermCanonUsps2cnt[str(usp_canonical_form)] += 1
                                    annoPretermNode2canon_usp[curr_node] = usp_canonical_form
                                else:
                                    preTermCanonUsps2cnt[str(uspCarg_canonical_form)] += 1
                                    annoPretermNode2canon_usp[curr_node] = usp_canonical_form
#                             dmrsSubgrCanon2bTag2cnt[str(canonical_form)][curr_derivRule] += 1
#                             dmrsSubgrCanonUsp2psdPreTerm2cnt[str(usp_canonical_form)]\
#                                 [util.get_coarse_preTerm(curr_derivRule)] += 1
#                             uRule = None
                            # update deriv 2 dmrs_repl mapping
                            annoDerivNode2dmrsNodeSubgr_repl[curr_node] = [curr_node * 100 + 20000]
                            dmrs_anchors2nodes_repl[curr_deriv_anchor] = [curr_node * 100 + 20000]
                        # e.g. both: card(carg:2) -1/EQ-> n <-RSTR/H- _both_q; imperative Don't bark!;
                        # [compound] -2/EQ-> named(Fund) <-1/EQ- [_firemans_a_1] (?)
                        # tentaively: just proceed
                        # TODO: bridged subgraph recognition
                        else:
                            canonical_form, sorted_nodes, _ = util.get_canonical_form(node_ind_subgraph_repl,
                                dmrs_nxDG_repl_orig,
                                extract_surfacemap = False)
                            uspCarg_canonical_form, sorted_nodes_uspCarg, _ = util.get_canonical_form(node_ind_subgraph_repl,
                                dmrs_nxDG_repl_orig, lexicalized = lexicalized,
                                extract_surfacemap = False, underspecCarg = True)
                            usp_canonical_form, sorted_nodes_usp, _ = util.get_canonical_form(node_ind_subgraph_repl,
                                dmrs_nxDG_repl_orig, lexicalized = lexicalized,
                                extract_surfacemap = False, underspecLemma = True, underspecCarg = True)
                            ruleToReplace = uspCarg_canonical_form
#                             if lexicalized:
#                                 if len(sorted_nodes) == 1:
#                                     ruleToReplace = uspCarg_canonical_form
                            interSubgrs_edges_lbls, interSubgrs_edges_key_orig = util.replace_subgraph(dmrs_nxDG_repl,
                                                                       dmrs_nxDG,
                                                                       annoDerivNode2dmrsNodeSubgr_repl[curr_node],
                                                                  [sorted_nodes_uspCarg],
                                                                    [True],
                                                                  ruleToReplace,
                                                                  curr_deriv_anchor,
                                                                  curr_node * 100 + 20000,
                                                              mode = mode, derivRule_usp = usp_canonical_form, node_typed = node_typed)
                            # update deriv 2 dmrs_repl mapping
                            annoDerivNode2dmrsNodeSubgr_repl[curr_node] = [curr_node * 100 + 20000]
                            dmrs_anchors2nodes_repl[curr_deriv_anchor] = [curr_node * 100 + 20000]
#                             print (curr_node, "Utag disconnected")
#                             print (canonical_form)
#                             #util.write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, sentence)
#                             input()
 
                    else:
                        print ("what case?")
                        input()
                        pass
#                     elif curr_node in annoDerivNode2dmrsNodeSubgr:
#                         surface = util.get_surface_of_derivSubTree(anno_deriv_nxDG_uni, curr_node)
#                         preTermCanon2surface2cnt[anno_deriv_nxDG_uni.nodes[curr_node]['entity']]\
#                                     [surface] += 1
#                         print ("Surface: ", surface)

                ## extract unary rule (lexical or syntactic)
                # first check if there's abstract pred associated to parent deriv node
                # e.g. generic_entity as in "Ben's barked".
                # if there is, check if neighbouring nodes of dgtr's dmrs node/subgraph is with curr_deriv_anchor
                # Currently BP will consume all ccont predicates, so no UP will be present
                elif curr_deriv_anchor in dmrs_anchors2nodes\
                        and all([not dmrs_node2isextracted[dmrs_node]
                                 for dmrs_node in dmrs_anchors2nodes[curr_deriv_anchor]]):
                    mode = "UP"
                    if print_debug:
                        print ("mode: {} on".format(mode), curr_node)
                    for dmrs_node in dmrs_anchors2nodes[curr_deriv_anchor]:
                        dmrs_node2isextracted[dmrs_node] = True
                    # record intragraph edges as well in case of multiple introduced nodes
                    for edge in curr_node_ind_subgraph.edges:
                        assert dmrs_edge2isextracted[edge] == False
                        dmrs_edge2isextracted[edge] = True
                    for edge in interSubgrCD_edges_key:
                        assert dmrs_edge2isextracted[edge] == False
                        dmrs_edge2isextracted[edge] = True
                        
                    annoDerivNode2dmrsNodeSubgr[curr_node] = list(set(dgtr_dmrs_nodes\
                        + dmrs_anchors2nodes[curr_deriv_anchor]))
                    annoDerivNode2dmrsNodeSubgr_repl[curr_node] = list(set(dgtr_dmrs_nodes_repl\
                        + dmrs_anchors2nodes_repl[curr_deriv_anchor]))
                    curr_node_ind_subgraph = dmrs_nxDG.subgraph(dmrs_anchors2nodes[curr_deriv_anchor])
                    curr_node_ind_subgraph_repl = dmrs_nxDG_repl.subgraph(dmrs_anchors2nodes_repl[curr_deriv_anchor])
                    # extract abstract pred's interaction with the daughter dmrs subgraphs; record it
                    interSubgrCD_edges_key, interSubgrCD_edges_lbl = util.get_interSubgr_edges(dmrs_nxDG,
                        dmrs_anchors2nodes[curr_deriv_anchor],
                        dgtr_dmrs_nodes, directed = False)
                    # print (interSubgr_edges)
                    # record interaction of introduced dmrs nodes with daughter dmrs subgraph 
                    if interSubgrCD_edges_key:
                        if is_weakly_connected(curr_node_ind_subgraph_repl):
                            canonical_form, sorted_nodes, _ = util.get_canonical_form(curr_node_ind_subgraph_repl,
                                dmrs_nxDG_repl_orig,
                                extract_surfacemap = False)
                            usp_canonical_form, sorted_nodes_usp, _ = util.get_canonical_form(curr_node_ind_subgraph_repl,
                                    dmrs_nxDG_repl_orig,
                                    extract_surfacemap = False, underspecLemma = True, underspecCarg = True, lexicalized = lexicalized)
                            uRule = dmrs_nxDG_repl.nodes[dgtr_dmrs_nodes_repl[0]]['derivRule_usp']
#                             uRule = util.get_coarse_preTerm(uRule)
#                             assert (uRule) == dgtr_derivRule
                            interSubgrs_edges_lbls, interSubgrs_edges_key_orig = util.replace_subgraph(dmrs_nxDG_repl,
                                                                               dmrs_nxDG,
                                                                          annoDerivNode2dmrsNodeSubgr_repl[curr_node],
                                                                [sorted_nodes, dgtr_dmrs_nodes_repl],
                                                                            [True, False],
                                                                          curr_derivRule,
                                                                          curr_deriv_anchor,
                                                                  curr_node * 100 + 20000,
                                                                      mode = mode, node_typed = node_typed)
                            # update deriv 2 dmrs_repl mapping
                            annoDerivNode2dmrsNodeSubgr_repl[curr_node] = [curr_node * 100 + 20000]
                            dmrs_anchors2nodes_repl[curr_deriv_anchor] = [curr_node * 100 + 20000]
                            # rules collection
                            canon2cnt[str(canonical_form)] += 1
                            canon_usp2cnt[str(usp_canonical_form)] += 1
                            SHRG[str(((canonical_form, uRule), interSubgrs_edges_lbls))][curr_derivRule] += 1
                            SHRG_coarse[str(((canonical_form, uRule), interSubgrs_edges_lbls))][curr_derivRule] += 1 
                            ccontCanon2intSubgrEdges2cnt[str(canonical_form)][str(interSubgrs_edges_lbls)] += 1
                            print ("UP")
#                             print (interSubgrs_edges_lbls)
#                             # handle semEmt for e.g. phrasal verb, copula
#                             # print ("look for trg")
#                             for node in dmrs_anchors2nodes[curr_deriv_anchor]:
#                                 semEmt = util.get_pred_prtcl(dmrs_nxDG.nodes[node]['instance'])
#                                 # copula trg
#                                 copula_trg, tense = util.get_copula_trg(dmrs_nxDG, node)
#                                 if (not anno_deriv_nxDG_uni.nodes[curr_node]['copula_trg']\
#                                     or tense != 'UNTENSED')\
#                                     and copula_trg:
#                                     anno_deriv_nxDG_uni.nodes[curr_node]['copula_trg'] = copula_trg
#                             if anno_deriv_nxDG_uni.nodes[curr_node]['copula_trg']:
#                                 print ("copula trg: ", dmrs_nxDG.nodes[node], copula_trg, mode)

                        # e.g. two unary rules anchored equally, each introducing different predicates
                        else:
                            canonical_form, sorted_nodes, _ = util.get_canonical_form(curr_node_ind_subgraph_repl,
                                dmrs_nxDG_repl_orig,
                                extract_surfacemap = False)
                            uRule = dmrs_nxDG_repl.nodes[dgtr_dmrs_nodes_repl[0]]['derivRule_usp']
                            assert (uRule) == dgtr_derivRule
                            interSubgrs_edges_lbls, interSubgrs_edges_key_orig = util.replace_subgraph(dmrs_nxDG_repl,
                                                                               dmrs_nxDG,
                                                                          annoDerivNode2dmrsNodeSubgr_repl[curr_node],
                                                                [sorted_nodes, dgtr_dmrs_nodes_repl],
                                                                            [True, False],
                                                                          curr_derivRule,
                                                                          curr_deriv_anchor,
                                                                  curr_node * 100 + 20000,
                                                                      mode = mode, node_typed = node_typed)
                            # update deriv 2 dmrs_repl mapping
                            annoDerivNode2dmrsNodeSubgr_repl[curr_node] = [curr_node * 100 + 20000]
                            dmrs_anchors2nodes_repl[curr_deriv_anchor] = [curr_node * 100 + 20000]
#                             print (curr_node, "UP introduced pred disconnected")
#                             print (canonical_form)
#                             #util.write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, sentence)
#                             input()
                        interSubgrEdgeNo2cnt[len(interSubgrCD_edges_key)] += 1
                            
                    else:
                        print ("no connection btwn UP P and dgtr")
                        print (curr_derivRule)
                        #util.write_figs_err(dmrs_nxDG_repl, None, curr_derivRule)
                        input()
                    
                     
                else:
                    mode = "U"
                    if print_debug:
                        print ("mode: {} on".format(mode), curr_node)
                    if dgtr in annoDerivNode2dmrsNodeSubgr:
                        annoDerivNode2dmrsNodeSubgr[curr_node] = annoDerivNode2dmrsNodeSubgr[dgtr]
                        annoDerivNode2dmrsNodeSubgr_repl[curr_node] = annoDerivNode2dmrsNodeSubgr_repl[dgtr]
                    # semantically empty lexeme, empty list 
                    else:
                        annoDerivNode2dmrsNodeSubgr[curr_node] = []
                        annoDerivNode2dmrsNodeSubgr_repl[curr_node] = []
#                     print (annoDerivNode2dmrsNodeSubgr_repl, curr_node)
                    uRule = dmrs_nxDG_repl.nodes[dgtr_dmrs_nodes_repl[0]]['derivRule_usp']
#                     uRule = util.get_coarse_preTerm(uRule)
                    # assert (uRule) == dgtr_derivRule
                    interSubgrs_edges_lbls, interSubgrs_edges_key_orig = util.replace_subgraph(dmrs_nxDG_repl,
                                                               dmrs_nxDG,
                                                                  annoDerivNode2dmrsNodeSubgr_repl[curr_node],
                                                               [annoDerivNode2dmrsNodeSubgr_repl[curr_node]],
                                                                [False],
                                                                  curr_derivRule,
                                                                  curr_deriv_anchor,
                                                          curr_node * 100 + 20000,
                                                              mode = mode, node_typed = node_typed)
                    # rules collection
                    SHRG[str(((uRule,), interSubgrs_edges_lbls))][curr_derivRule] += 1
                    SHRG_coarse[str(((uRule,), interSubgrs_edges_lbls))][curr_derivRule] += 1 
                    # update deriv 2 dmrs_repl mapping
                    annoDerivNode2dmrsNodeSubgr_repl[curr_node] = [curr_node * 100 + 20000]
                    dmrs_anchors2nodes_repl[curr_deriv_anchor] = [curr_node * 100 + 20000]
                if uRule:
                    dgtrs2unaryRule2cnt[str(uRule)][curr_derivRule] += 1 
                    ergRule2cnt[curr_derivRule] += 1
#             pprint (SHRG)
#             print ()
            anno_deriv_nxDG_uni.nodes[curr_node]['extracted'] = True
        
        
#         # propagate sematically empty lexical entry info to parent node
#         # i.e. if any dgtr contains sem-emt related prop, propagate to curr_node
#         semEmtTrgAtlorR = util.propagate_semEmtInfo_in_deriv(anno_deriv_nxDG_uni, curr_node, semEmtType = "semEmt")
#         # print (curr_node, "propagate error")
        # util.write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, None, None)
#         # input()
#         copTrgAtlorR = util.propagate_semEmtInfo_in_deriv(anno_deriv_nxDG_uni, curr_node, semEmtType = "copula")    

    dmrs_anchors2nodes, dmrs_anchors2preds = util.get_equalanchor_nodes(dmrs_nxDG, lexical_only = False)
    dmrs_anchors2nodes_repl = dmrs_anchors2nodes.copy()
    
    anno_deriv_nxDG_uni = anno_deriv_nxDG.copy()
    dmrs_extractedNodes = set() 
#     dmrs_extractedNodes = reduce(lambda x, y: set(x).union(set(y)), annoDerivPreTerm2dmrsNodeSubgr.values()) 
    dmrs_node2isextracted = {node: False if node not in dmrs_extractedNodes else True
                             for node in dmrs_nxDG.nodes()}
    dmrs_edge2isextracted = {edge: False for edge in dmrs_nxDG.edges}
#     #util.write_figs_err(dmrs_nxDG, anno_deriv_nxDG_uni, sentence)
#     pprint (dmrs_node2isextracted)
    dmrs_nxDG_repl = dmrs_nxDG.copy()
    # augment edge labels for external node recognition
    for edge in dmrs_nxDG_repl.edges:
        if node_typed:
            _, src_pos = util.get_lemma_pos(dmrs_nxDG.nodes[edge[0]])
            _, targ_pos = util.get_lemma_pos(dmrs_nxDG.nodes[edge[1]])
            src_pos = util.unknown2pos.get(src_pos) or src_pos
            targ_pos = util.unknown2pos.get(targ_pos) or targ_pos
            dmrs_nxDG_repl.edges[edge]['label'] = '#0' + src_pos + '-src-' + dmrs_nxDG_repl.edges[edge]['label']\
                + '-targ-'+ targ_pos + '#0'

        else:
            dmrs_nxDG_repl.edges[edge]['label'] = '#0-src-' + dmrs_nxDG_repl.edges[edge]['label'] + '-targ-#0'
    # assign ordered ext nodes to itself
    for node_repl in dmrs_nxDG_repl.nodes:
        dmrs_nxDG_repl.nodes[node_repl]['ordered_ext_nodes'] = str(node_repl)
    dmrs_nxDG_repl_orig = dmrs_nxDG_repl.copy()
    for node, node_prop in anno_deriv_nxDG_uni.nodes(data = True):
        if not 'cat' in node_prop: continue
        anno_deriv_nxDG_uni.nodes[node]['comma'] = False
        if node_prop['cat'] != '<terminal>':
            anno_deriv_nxDG_uni.nodes[node]['extracted'] = False
        else:
            anno_deriv_nxDG_uni.nodes[node]['extracted'] = True
            if node_prop['form'][-1] == ',':
                anno_deriv_nxDG_uni.nodes[node]['comma'] = True
        for semEmt_attr in ["prtcl_trg", "copula_trg", "compl_trg"]:
            anno_deriv_nxDG_uni.nodes[node][semEmt_attr] = None
        for semEmt_attr in ["prtcl", "copula", "compl", 'by', "predSemEmt"]:
            anno_deriv_nxDG_uni.nodes[node][semEmt_attr] = []
   
            
    annoDerivNode2dmrsNodeSubgr = annoDerivPreTerm2dmrsNodeSubgr
    annoDerivNode2dmrsNodeSubgr_repl = annoDerivNode2dmrsNodeSubgr.copy()
    old_annoDerivNode2dmrsNodeSubgr = annoDerivNode2dmrsNodeSubgr.copy()
    annoPretermNode2canon_usp = defaultdict()
    
#     util.write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, sentence, sentence.translate(str.maketrans('', '', string.punctuation)))
#     print (anno_deriv_nxDG_uni.nodes[anno_deriv_nxDG_uni.graph['root']]['entity'])
    startRule2cnt[anno_deriv_nxDG_uni.nodes[anno_deriv_nxDG_uni.graph['root']]['entity']] += 1
    _dfs_extract(annoDerivNode2dmrsNodeSubgr, annoDerivNode2dmrsNodeSubgr_repl, annoPretermNode2canon_usp, dmrs_node2isextracted, dmrs_edge2isextracted, dmrs_nxDG, dmrs_nxDG_repl, dmrs_nxDG_repl_orig, dmrs_nxDG_ancAdj, anno_deriv_nxDG_uni, anno_deriv_nxDG_uni.graph['root'])
#     util.write_figs_err(None, anno_deriv_nxDG_uni, sentence, "final"+sentence[:50])
#     print (sentence)
    # assert True orion after finishing rule extraction
    # e.g. terminal subgraph with more than 6 nodes
    if len(dmrs_nxDG_repl.nodes) != 1:
#         print (sentence, ": not fully extracted")
        pass
        # #util.write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, sentence)
    
    
    # e.g. two equal-anchored preds introduced separately for different unary rule
    if any([dmrs_edge2isextracted[edge] != True for edge in dmrs_edge2isextracted]):
#         print (snt_id)
#         pprint (dmrs_edge2isextracted)
#         pprint (old_annoDerivNode2dmrsNodeSubgr)
#         pprint (annoDerivNode2dmrsNodeSubgr)
#         #util.write_figs_err(dmrs_nxDG, anno_deriv_nxDG, sentence)
#         input()
        pass
    
    return None
    
def extract_rules(training_sntid2item, trainOn = None, wsjOnly = False, stopAfter = 2147483629):
    # extract rules:
    # 1. pcfg of anno_deriv
    # 2. synchronous grammar of <dmrs_graph (where nodes/subgraphs are substituted by
    # erg construction w/ (some?) props), cfg rule/subtree(e.g. aux verb) of erg>
    # 3. edge collapse order of each dmrs node(/subgraph?)
    # 4. particle attachment time of each pred x {act, pass} (in terms of edge collapsing time
    # 4.5 copula/aux verb, to-inf
    # 4.75 frequent semantically empty lexeme, e.g. than, by (parg_d), ...
    # 4.999 be handled by chart generation-like process?
    # - 1. instantiate sem-empty lexeme
    # - 2. includ
    # 5. ltop and index of subgraphs
    # 6. fallbacks for every type of rules for 100% coverage
    # e.g. generalize deriv nodes that correspond to a named dmrs node
    # 7. two ver? one w/ one w/out carg
    semEmtLexEnt = Counter()
    gigaword_cnt = 0
    wsj_cnt = 0
    for idx, snt_id in enumerate(tqdm(training_sntid2item)):
        if snt_id[:3] == 'wsj':
            wsj_cnt += 1
    print ("#wsj training instance:", wsj_cnt)
        
    
    for idx, snt_id in enumerate(tqdm(training_sntid2item)):
        # print (idx)
        if idx < 58167:
            pass
        if wsjOnly:
            if snt_id[:3] != 'wsj': continue
        if trainOn != None:
            if idx != trainOn:
                continue
        #if snt_id != 'jhu/3035811':
            #continue
        if snt_id.startswith('gigaword'):
            gigaword_cnt += 1
            if gigaword_cnt > stopAfter:
                continue
        
#         if idx > 50: continue
        dmrs_nodelink_dict = training_sntid2item[snt_id]["dmrs_nodelinkdict"]
        dmrs_nxDG = nx.readwrite.json_graph.node_link_graph(dmrs_nodelink_dict)
        dmrs_ancAdj_nodelink_dict = training_sntid2item[snt_id]["ancAdj_dmrs_nodelinkdict"]
        dmrs_nxDG_ancAdj = nx.readwrite.json_graph.node_link_graph(dmrs_ancAdj_nodelink_dict)
        anno_derivation_nodelink_dict = training_sntid2item[snt_id]["anno_derivation_nodelinkdict"]
        anno_deriv_nxDG = nx.readwrite.json_graph.node_link_graph(anno_derivation_nodelink_dict)
        sentence = training_sntid2item[snt_id]["sentence"]
        # print (sentence)
        found = False
#         for node, nodes_prop in anno_deriv_nxDG.nodes(data = True):
#             if nodes_prop.get('entity') in ['out_particle']:
#                 found = True
#         if not found:
#             continue
#         print (idx, snt_id, sentence)
        if sentence != 'Hand Abrams money to buy the dog.':
            pass
        
#         util.write_figs_err(dmrs_nxDG, anno_deriv_nxDG, sentence)
#         input()
        # align dmrs node to deriv node
        
        annoDerivNode2dmrsNodes, semEmtLexEnt_tmp =\
            align_dmrs_to_annoderiv(snt_id, dmrs_nxDG, anno_deriv_nxDG, sentence)
        semEmtLexEnt += semEmtLexEnt_tmp
#             get_node2
        # obtain syn. grammar; save rules in global variables
#         print (idx, snt_id, sentence)
        extract_sync_grammar(snt_id, annoDerivNode2dmrsNodes, dmrs_nxDG, dmrs_nxDG_ancAdj,
                                 anno_deriv_nxDG, sentence, semEmtLexEnt_tmp)
        
#         util.write_figs_err(dmrs_nxDG, anno_deriv_nxDG, sentence)
#         input()
#         for canon in dmrsNodeSubgr2preTerm_tmp:
#             dmrsNodeSubgr2preTerm[canon] += dmrsNodeSubgr2preTerm_tmp[canon]
#         interSubgrEdgeNo2cnt += interSubgrEdgeNo2cnt
#         binaryRuleWithPreds_predLeftRightEdges +=  
#         subgraph2derivRule_list += dmrsSubgrCanon2bTag2cnt
#     print (interSubgrEdgeNo2cnt)
#     print (binaryRuleWithPreds_predLeftRightEdges)
#     pprint (ccontCanon2intSubgrEdges2cnt)
#     print ("=====")
#     with open("extracted_rules/combined-standard/all/startRule2cnt.json", "w") as f:
#         json.dump(startRule2cnt, f)
#     input()
    return semEmtLexEnt #dmrsNodeSubgr2preTerm, subgraph2derivRule_list

def estimate_phrg(extracted_file2data):
    SHRG_coarse = extracted_file2data['SHRG_coarse']
    PHRG = defaultdict(defaultdict)
    par2cnt = Counter()
    for shrg_key, _par2cnt in SHRG_coarse.items():
        par2cnt += _par2cnt
    for shrg_key, _par2cnt in SHRG_coarse.items():
        for par, cnt in _par2cnt.items():
            PHRG[shrg_key][par] = math.log(cnt/par2cnt[par])
    return PHRG

def estimate_pcfg(extracted_file2data):
    dgtrs2unaryRule2logProb = defaultdict(Counter)
    dgtrs2binaryRule2logProb = defaultdict(Counter)
    
    dgtrs2unaryRule2prob = defaultdict(Counter)
    dgtrs2binaryRule2prob = defaultdict(Counter)
    startRule2cnt, ergRule2cnt, dmrsSubgrCanon2bTag2cnt, preTermCanon2bTagSurface2cnt, preTerm2surface2cnt, preTermAnno2surface2cnt, preTermCanon2surface2cnt, preTermCanonUsps2cnt, dgtrs2unaryRule2cnt, dgtrs2binaryRule2cnt, SHRG, SHRG_coarse, ccontCanon2intSubgrEdges2cnt, dgtrsEdgesTrg2SubtreeDgtrs, canon2cnt, canon_usp2cnt\
        = [extracted_file2data[key] for key in "startRule2cnt ergRule2cnt dmrsSubgrCanon2bTag2cnt preTermCanon2bTagSurface2cnt preTerm2surface2cnt preTermAnno2surface2cnt preTermCanon2surface2cnt preTermCanonUsps2cnt dgtrs2unaryRule2cnt dgtrs2binaryRule2cnt SHRG SHRG_coarse ccontCanon2intSubgrEdges2cnt dgtrsEdgesTrg2SubtreeDgtrs canon2cnt canon_usp2cnt".split(" ")]
    check_sum1 = defaultdict(lambda: 0)
    for dgtrs in dgtrs2unaryRule2cnt:
        for unary_rule in dgtrs2unaryRule2cnt[dgtrs]:
            denom = ergRule2cnt[unary_rule]
            numer = dgtrs2unaryRule2cnt[dgtrs][unary_rule]
            prob = numer/denom
            log_prob = math.log(prob)
            assert log_prob <= 0
            dgtrs2unaryRule2prob[dgtrs][unary_rule] = prob
            dgtrs2unaryRule2logProb[dgtrs][unary_rule] = log_prob
            check_sum1[unary_rule] += prob
#     for s in check_sum1:
#         assert math.isclose(check_sum1[s] , 1)
#     check_sum1 = defaultdict(lambda: 0)
    for dgtrs in dgtrs2binaryRule2cnt:
        for binary_rule in dgtrs2binaryRule2cnt[dgtrs]:
            denom = ergRule2cnt[binary_rule]
            numer = dgtrs2binaryRule2cnt[dgtrs][binary_rule]
            prob = numer/denom
            log_prob = math.log(prob)
            assert log_prob <= 0
            dgtrs2binaryRule2prob[dgtrs][binary_rule] = prob
            dgtrs2binaryRule2logProb[dgtrs][binary_rule] = log_prob
            check_sum1[binary_rule] += prob
    for s in check_sum1:
        try:
            assert math.isclose(check_sum1[s] , 1)
        except:
            print (s, check_sum1[s], "not sum to 1 log prob")
    return dgtrs2unaryRule2logProb, dgtrs2binaryRule2logProb

def get_equalAnc_semiCanons(preTermCanon2surface2cnt, underspecLemma = False, keeplen1 = False):
    eqAnc_semiCanons = set()
    for canon in preTermCanon2surface2cnt:
#         print (canon)
        canon = make_tuple(canon) 
        if len(canon) == 1 and keeplen1:
            continue
        subgraph_preds = list()
        for nodeEdgeRep in canon:
            nodeRep = nodeEdgeRep[0]
#             try:
            pred = nodeRep.split("ep:")[1].split(";")[0]
#             print (pred)
            subgraph_preds.append(pred)
#             except:
#                 print (nodeRep, nodeEdgeRep)
        semiCanon = util.get_semicanonical_form(subgraph_preds, underspecLemma)
        eqAnc_semiCanons.add(semiCanon)
    return eqAnc_semiCanons

def get_semEmtsubtree_logProb(dgtrsEdgesTrg2SubtreeDgtrs,
                              dgtrs2unaryRule2logProb, dgtrs2binaryRule2logProb):
#     dgtrsEdgesTrg2SubtreeDgtrs_copula
    has_logProb = True
    def _dfs_logProb(subtree, left_dgtr, right_dgtr):
#         print (subtree)
        nonlocal has_logProb
        curr_node, curr_ent, curr_children = subtree['id'], subtree['entity'], subtree.get("children")
        if not curr_children:
            if not curr_node in [left_dgtr, right_dgtr]:
                return 0, [curr_ent]
            elif curr_node == left_dgtr:
                return 0, ['<left>']
            elif curr_node == right_dgtr:
                return 0, ['<right>']
        else:
            if len(curr_children) == 2:
                ld, rd = curr_children
                ld_ent, rd_ent = ld['entity'], rd['entity']
                l_logProb, l_preterms = _dfs_logProb(ld, left_dgtr, right_dgtr)
                r_logProb, r_preterms = _dfs_logProb(rd, left_dgtr, right_dgtr)
#                 if str((ld_ent, rd_ent))
                if str((ld_ent, rd_ent)) not in dgtrs2binaryRule2logProb:
                    print ("key not found:", str((ld_ent, rd_ent)))
                    has_logProb = False
                    raise Exception("key not found:" + str((ld_ent, rd_ent)))
                else:
                    return dgtrs2binaryRule2logProb[str((ld_ent, rd_ent))][curr_ent]\
                            + l_logProb + r_logProb, l_preterms + r_preterms
#                 except:
#                     print ((ld_ent, rd_ent), "not recorded in logProbs!")
            elif len(curr_children) == 1:
                ud = curr_children[0]
                ud_ent = ud['entity']
                logProb, preterms = _dfs_logProb(ud, left_dgtr, right_dgtr)
                return logProb + dgtrs2unaryRule2logProb[ud_ent][curr_ent], preterms
            
    def rm_subtree_nodeid(subtree):
        curr_subtree = {}
        _, curr_ent, curr_children = subtree['id'], subtree['entity'], subtree.get("children")
        curr_subtree['entity'] = curr_ent
        if curr_children:
#             print (curr_children)
            curr_subtree['chlidren'] = [rm_subtree_nodeid(c) for c in curr_children]
        return curr_subtree
        
#     for a in dgtrsEdgesTrg2SubtreeDgtrs['prtcl']:
#         print (a)
#         input()
    dgtrsTrgs2edges2surf2logProb = defaultdict(lambda: defaultdict(lambda: defaultdict(defaultdict)))
    for semEmt_type in  ['copula', 'prtcl', 'compl', 'by', 'predSemEmt']:
        preterms_set = set()
        ruleEdge2cnt2 = Counter()
        for key_str, subtrees in dgtrsEdgesTrg2SubtreeDgtrs[semEmt_type].items():
            ruleTrgs, interSubgrs_edges_lbls\
                = make_tuple(key_str)
#             if len(ruleTrgs) == 2:
#                 (left_cfg_dgtr_ent, semEmtTrgL), (right_cfg_dgtr_ent, semEmtTrgR) = ruleTrgs
#             elif len(ruleTrgs) == 3:
#                 (ccont_ent ,_), (left_cfg_dgtr_ent, semEmtTrgL), (right_cfg_dgtr_ent, semEmtTrgR) = ruleTrgs
            # choose one subtree for each <dgtrs, edges>
            idDel_subtrees2cnt = Counter()
            idDel2subtree = defaultdict()
            for subtree_str, cnt in subtrees.items():
                subtree, left_dgtr, right_dgtr = make_tuple(subtree_str)
                idDel_subtree = rm_subtree_nodeid(subtree)
                idDel_subtrees2cnt[str(idDel_subtree)] += 1
                if str(idDel_subtree) not in idDel2subtree:
                    idDel2subtree[str(idDel_subtree)] = (subtree, left_dgtr, right_dgtr)
            sorted_subtree = sorted(idDel_subtrees2cnt.items(), key = lambda x:x[1], reverse = True)
            total_cnt = sum(list(zip(*sorted_subtree))[1])
            #if semEmt_type == 'copula' and total_cnt <= 1:
            #    continue
            ruleEdge2cnt2[total_cnt] += 1
            for subtree_str, cnt in sorted_subtree:
#                 if cnt <= 1: continue
                subtree, left_dgtr, right_dgtr = idDel2subtree[subtree_str]
                curr_derivRule = subtree['entity']
#                 try:
                has_logProb = True
                try:
                    logProbPreterms = _dfs_logProb(subtree, left_dgtr, right_dgtr)
                except:
                    continue
                if logProbPreterms and has_logProb:
                    logProb, preterms = logProbPreterms
                    if len(preterms) <= 2: continue
                    nice_rule = True
                    if semEmt_type == 'prtcl': nice_rule = False
                    for p in preterms:
                        if semEmt_type == 'copula':
                            if "_particle" in p or "_prtcl" in p or 'that_c' in p:
                                nice_rule = False
                                break
                        elif semEmt_type == 'prtcl':
                            if "_particle" in p or "_prtcl" in p:
                                nice_rule = True
                                break
                        if "_disc" in p or "however" in p or "_filler" in p or "&" in p:
                            nice_rule = False
                            break
                            
#                     invalid = False
#                     for p in preterms:
#                         if "_prtcl" in p or "_particle" in p:
#                             invalid = True
#                             break
#                     if invalid: continue
                    if nice_rule:
                        normalized_preterms = tuple(p.split("&")[0].split("/")[0] for p in preterms)
                        preterms_set.add(tuple(normalized_preterms))
                        dgtrsTrgs2edges2surf2logProb[semEmt_type][str(ruleTrgs)]\
                            [str(interSubgrs_edges_lbls)][str((curr_derivRule, tuple(normalized_preterms)))]\
                            = max(logProb, dgtrsTrgs2edges2surf2logProb[semEmt_type][str(ruleTrgs)]\
                                  [str(interSubgrs_edges_lbls)].get(str((curr_derivRule, tuple(normalized_preterms)))) or -2147483629)
                    break
#                 except:
#                     print (subtree, left_dgtr, right_dgtr, ": some dgtrs key not in logProb dicts; not included")
    #                 continue
        print (semEmt_type)
        print (ruleEdge2cnt2)
        pprint (preterms_set)
        print ("------")

    return dgtrsTrgs2edges2surf2logProb
            
        
def main(annotatedData_dir, extracted_rules_dir, redwdOrGw, config_name, sampleOrAll, nodeTyped = False, stopAfter = 2147483629, wsjOnly = False, lexicalize = False, trainOrTest = "train", trainOn = None):
    
    global startRule2cnt, ergRule2cnt, dmrsSubgrCanon2bTag2cnt, preTermCanon2bTagSurface2cnt, preTerm2surface2cnt, preTermAnno2surface2cnt, preTermCanon2surface2cnt, preTermCanonUsps2cnt, dgtrs2unaryRule2cnt, dgtrs2binaryRule2cnt, SHRG, SHRG_coarse, ccontCanon2intSubgrEdges2cnt, dgtrsEdgesTrg2SubtreeDgtrs, canon2cnt, canon_usp2cnt, eqAnc_semiCanons, eqAnc_semiCanons_usp, ccont_semiCanons #interSubgrEdgeNo2cnt
    
    startRule2cnt = Counter()
    ergRule2cnt = Counter()
    dmrsSubgrCanon2bTag2cnt = defaultdict(Counter)
    preTermCanon2bTagSurface2cnt = defaultdict(Counter)
    preTerm2surface2cnt = defaultdict(Counter)
    preTermAnno2surface2cnt = defaultdict(Counter)
    preTermCanon2surface2cnt = defaultdict(Counter)
    preTermCanonUsps2cnt = Counter() 
    dgtrs2unaryRule2cnt = defaultdict(Counter)
    dgtrs2binaryRule2cnt = defaultdict(Counter)
    canon2cnt = Counter()
    canon_usp2cnt = Counter()
    SHRG = defaultdict(Counter)
    SHRG_coarse = defaultdict(Counter)
    ccontCanon2intSubgrEdges2cnt = defaultdict(Counter)
    interSubgrEdgeNo2cnt = Counter()
    dgtrsEdgesTrg2SubtreeDgtrs = defaultdict(lambda: defaultdict(Counter))
    eqAnc_semiCanons = set()
    eqAnc_semiCanons_usp = set()
    ccont_semiCanons = set()
    
    extracted_file2data = get_extractedFileName2data()
    
    sys.stderr.write("Loading config file ...\n")
    config, config_filepath = util.load_config(config_name, redwdOrGw)
    
    # pprint (config)
    global max_subgraph_size, lexicalized, node_typed
    max_subgraph_size = config['train']['max_subgraph_size']
    lexicalized = lexicalize
    node_typed = nodeTyped
     
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

    if trainOrTest == 'train':
        sys.stderr.write('Training...\n')
        sys.stderr.write("Loading annotated training data ...\n")
        with open(training_data_filepath, "r", encoding='utf-8') as f:
            training_data = json.load(f)
            print (len(training_data))
    elif trainOrTest == 'test':
        sys.stderr.write('Testing...\n')
        sys.stderr.write("Loading annotated test data ...\n")
        with open(test_data_filepath, "r", encoding='utf-8') as f:
            training_data = json.load(f)
            
    sys.stderr.write("Loaded!\n")
    
    extracted_rules_dir += "-28102021"
    extracted_rules_conf_dir = os.path.join(extracted_rules_dir, redwdOrGw + '-'\
                                    + config_name) ####
    if wsjOnly:
        extracted_rules_conf_dir += "-wsj"
    if lexicalize:
        extracted_rules_conf_dir += "-lex"
    if nodeTyped:
        extracted_rules_conf_dir += "-nodeTyped"
    if trainOrTest == 'test':
        extracted_rules_conf_dir += "-test"
        
    extracted_rules_confSA_dir = os.path.join(extracted_rules_conf_dir, sampleOrAll) #####
    os.makedirs(extracted_rules_confSA_dir, exist_ok=True)
    
    sys.stderr.write('Extract rules now. This should take some time\n')
#     sys.stderr.write('Extract nodes to be lexicalized\m')
#     pred_lex
    
    
    sys.stderr.write('For every snt, align DMRS subgraphs/nodes to annotated derivation nodes, and SHRG is extracted\n')
    semEmtLexEnt = extract_rules(training_data, trainOn, wsjOnly, stopAfter) # dmrsNodeSubgr2preTerm
    sys.stderr.write('Done!\n')  
    
    sys.stderr.write('Dump comma prob.\n')
    rule2comma_prob = {rule: rule2comma[rule] / (rule2comma[rule] + rule2noComma[rule]) for rule in rule2comma}
    commaProb_path = os.path.join(extracted_rules_confSA_dir, "rule2commaProb" + ".json") ####
    with open(commaProb_path, "w") as f:
        json.dump(rule2comma_prob, f, indent=4)
    sys.stderr.write('Done!\n')  
    
    extracted_file2data = get_extractedFileName2data()
    
    sys.stderr.write('Start writing rules to file!\n')
    for name, data in extracted_file2data.items():
        extracted_rules_confSA_path = os.path.join(extracted_rules_confSA_dir, name + ".json")
        try:
            with open(extracted_rules_confSA_path, "w") as f:
                json.dump(data, f)
        except:
            try:
                with open(extracted_rules_confSA_path, "w") as f:
                    json.dump(list(data), f)
            except:
                print (name)
    sys.stderr.write("All rules dumped!")
    
    if not extracted_file2data["ergRule2cnt"]: #####
        for name in extracted_file2data: 
            extracted_rules_confSA_path = os.path.join(extracted_rules_confSA_dir, name + ".json")
            with open(extracted_rules_confSA_path, "r") as f:
                extracted_file2data[name] = json.load(f) #####
    
    sys.stderr.write('Start estimating PCFG/PHRG from rule file!\n')
    dgtrs2unaryRule2logProb, dgtrs2binaryRule2logProb = estimate_pcfg(extracted_file2data) # dmrsNodeSubgr2preTerm
    phrg = estimate_phrg(extracted_file2data)
    sys.stderr.write('Done!\n')
    sys.stderr.write('Start writing PCFG/PHRG to file!\n')

    unary_pcfg_path = os.path.join(extracted_rules_confSA_dir, "dgtrs2unaryRule2logProb" + ".json") ####
    binary_pcfg_path = os.path.join(extracted_rules_confSA_dir, "dgtrs2binaryRule2logProb" + ".json") ###
    phrg_path = os.path.join(extracted_rules_confSA_dir, "PHRG" + ".json") ###
    
    with open(unary_pcfg_path, "w") as f:
        json.dump(dgtrs2unaryRule2logProb, f)
    with open(binary_pcfg_path, "w") as f:
        json.dump(dgtrs2binaryRule2logProb, f)
    with open(phrg_path, "w") as f:
        json.dump(phrg, f)
    sys.stderr.write("SHRG/PHRG log prob dumped!\n")
    
    with open(unary_pcfg_path, "r") as f: ####
        dgtrs2unaryRule2logProb = json.load(f)
    with open(binary_pcfg_path, "r") as f:
        dgtrs2binaryRule2logProb = json.load(f) ####
        
    sys.stderr.write("PCFG log prob loaded!\n") #####
    dgtrsTrgs2edges2surf2logProb =\
        get_semEmtsubtree_logProb(extracted_file2data["dgtrsEdgesTrg2SubtreeDgtrs"],
                              dgtrs2unaryRule2logProb, dgtrs2binaryRule2logProb) #####
    sys.stderr.write("Augmented copula/semEmt dicts with logProb!\n") #####
    
#     dgtrsTrgs2edges2surf2logProb_copula, dgtrsTrgs2edges2surf2logProb_prtcl\
#         = [dgtrsTrgs2edges2surf2logProb[semEmt] for semEmt in ['copula', 'prtcl']]
    
    aug_copula_path = os.path.join(extracted_rules_confSA_dir, "dgtrsTrgs2edges2surf2logProb_copula" + ".json") ####
    aug_prtcl_path = os.path.join(extracted_rules_confSA_dir, "dgtrsTrgs2edges2surf2logProb_prtcl" + ".json") ###
    aug_semEmt_path = os.path.join(extracted_rules_confSA_dir, "dgtrsTrgs2edges2surf2logProb" + ".json") ### 
    
    with open(aug_semEmt_path, "w") as f: ####
        json.dump(dgtrsTrgs2edges2surf2logProb, f)
#     with open(aug_copula_path, "w") as f: ####
#         json.dump(dgtrsTrgs2edges2surf2logProb_copula, f)
#     with open(aug_prtcl_path, "w") as f: ####
#         json.dump(dgtrsTrgs2edges2surf2logProb_prtcl, f)
    sys.stderr.write("Dumped!\n") #####
    
    
#     sys.stderr.write('Extracting semi-canonical form of connected equal anchored DMRS subgraphs\n')
#     preTermCanon2surface2cnt = extracted_file2data["preTermCanon2surface2cnt"]
#     ccontCanon2intSubgrEdges2cnt = extracted_file2data["ccontCanon2intSubgrEdges2cnt"]
#     eqAnc_semiCanons = get_equalAnc_semiCanons(preTermCanon2surface2cnt)
#     eqAnc_semiCanons_usp = get_equalAnc_semiCanons(preTermCanonUsps2cnt)
#     ccont_semiCanons = get_equalAnc_semiCanons(ccontCanon2intSubgrEdges2cnt, keeplen1 = True)
#     sys.stderr.write('Done\n')
#     sys.stderr.write('Writing semi-canonical forms to files\n')
#     eqAnc_semiCanons_path = os.path.join(extracted_rules_confSA_dir, "eqAnc_semiCanons" + ".json")
#     eqAnc_semiCanons_usp_path = os.path.join(extracted_rules_confSA_dir, "eqAnc_semiCanons_usp" + ".json")
#     ccont_semiCanons_path = os.path.join(extracted_rules_confSA_dir, "ccont_semiCanons" + ".json")
#     with open(eqAnc_semiCanons_path, "w") as f: 
#         json.dump(list(eqAnc_semiCanons), f)
#     with open(eqAnc_semiCanons_usp_path, "w") as f:
#         json.dump(list(eqAnc_semiCanons_usp), f)
#     with open(ccont_semiCanons_path, "w") as f:
#         json.dump(list(ccont_semiCanons), f)
    sys.stderr.write('Done\n')
    
    return semEmtLexEnt  
    
def main2(annotatedData_dir, extracted_rules_dir, redwdOrGw, config_name, sampleOrAll, lexicalize = False, trainOrTest = "train", trainOn = None):
    
    global startRule2cnt, ergRule2cnt, dmrsSubgrCanon2bTag2cnt, preTermCanon2bTagSurface2cnt, preTerm2surface2cnt, preTermAnno2surface2cnt, preTermCanon2surface2cnt, preTermCanonUsps2cnt, dgtrs2unaryRule2cnt, dgtrs2binaryRule2cnt, SHRG, SHRG_coarse, ccontCanon2intSubgrEdges2cnt, dgtrsEdgesTrg2SubtreeDgtrs, canon2cnt, canon_usp2cnt, eqAnc_semiCanons, eqAnc_semiCanons_usp, ccont_semiCanons, rule2comma, rule2noComma #interSubgrEdgeNo2cnt
    
    startRule2cnt = Counter()
    ergRule2cnt = Counter()
    dmrsSubgrCanon2bTag2cnt = defaultdict(Counter)
    preTermCanon2bTagSurface2cnt = defaultdict(Counter)
    preTerm2surface2cnt = defaultdict(Counter)
    preTermAnno2surface2cnt = defaultdict(Counter)
    preTermCanon2surface2cnt = defaultdict(Counter)
    preTermCanonUsps2cnt = Counter() 
    dgtrs2unaryRule2cnt = defaultdict(Counter)
    dgtrs2binaryRule2cnt = defaultdict(Counter)
    canon2cnt = Counter()
    canon_usp2cnt = Counter()
    SHRG = defaultdict(Counter)
    SHRG_coarse = defaultdict(Counter)
    ccontCanon2intSubgrEdges2cnt = defaultdict(Counter)
    interSubgrEdgeNo2cnt = Counter()
    dgtrsEdgesTrg2SubtreeDgtrs = defaultdict(lambda: defaultdict(Counter))
    eqAnc_semiCanons = set()
    eqAnc_semiCanons_usp = set()
    ccont_semiCanons = set()
    rule2comma = Counter()
    rule2noComma = Counter()
    
    extracted_file2data = get_extractedFileName2data()
    
    sys.stderr.write("Loading config file ...\n")
    config, config_filepath = util.load_config(config_name, redwdOrGw)
    
    # pprint (config)
    global max_subgraph_size, lexicalized
    max_subgraph_size = config['train']['max_subgraph_size']
    lexicalized = lexicalize
     
#     all_training_data_filepath = os.path.join(annotatedData_dir, "training", "training_sntid2item.json")
#     all_dev_data_filepath = os.path.join(annotatedData_dir, "dev", "dev_sntid2item.json")
#     all_test_data_filepath = os.path.join(annotatedData_dir, "test", "test_sntid2item.json")
#     sample_training_data_filepath = os.path.join(annotatedData_dir, "training", "sample_training_sntid2item.json")
#     sample_dev_data_filepath = os.path.join(annotatedData_dir, "dev", "sample_dev_sntid2item.json")
#     sample_test_data_filepath = os.path.join(annotatedData_dir, "test", "sample_test_sntid2item.json")
    
#     if sampleOrAll == 'sample':
#         training_data_filepath = sample_training_data_filepath
#         dev_data_filepath = sample_dev_data_filepath
#         test_data_filepath = sample_test_data_filepath
#     elif sampleOrAll == 'all':
#         training_data_filepath = all_training_data_filepath
#         dev_data_filepath = all_dev_data_filepath
#         test_data_filepath = all_test_data_filepath

#     if trainOrTest == 'train':
#         sys.stderr.write('Training...\n')
#         sys.stderr.write("Loading annotated training data ...\n")
#         with open(training_data_filepath, "r", encoding='utf-8') as f:
#             training_data = json.load(f)
#             print (len(training_data))
#     elif trainOrTest == 'test':
#         sys.stderr.write('Testing...\n')
#         sys.stderr.write("Loading annotated test data ...\n")
#         with open(test_data_filepath, "r", encoding='utf-8') as f:
#             training_data = json.load(f)
            
#     sys.stderr.write("Loaded!\n")
    
    extracted_rules_dir += "-28102021"
    extracted_rules_conf_dir = os.path.join(extracted_rules_dir, redwdOrGw + '-'\
                                    + config_name) ####
    if lexicalize:
        extracted_rules_conf_dir += "-lex"
    if nodeTyped:
        extracted_rules_conf_dir += "-nodeTyped"
    if trainOrTest == 'test':
        extracted_rules_conf_dir += "-test"
    extracted_rules_confSA_dir = os.path.join(extracted_rules_conf_dir, sampleOrAll) #####
    os.makedirs(extracted_rules_confSA_dir, exist_ok=True)
    
#     sys.stderr.write('Extract rules now. This should take some time\n')
#     sys.stderr.write('For every snt, align DMRS subgraphs/nodes to annotated derivation nodes, and SHRG is extractedn')
#     semEmtLexEnt = extract_rules(training_data, trainOn) # dmrsNodeSubgr2preTerm
#     sys.stderr.write('Done!\n')
    extracted_file2data = get_extractedFileName2data()
    
#     sys.stderr.write('Start writing rules to file!\n')
#     for name, data in extracted_file2data.items():
#         extracted_rules_confSA_path = os.path.join(extracted_rules_confSA_dir, name + ".json")
#         try:
#             with open(extracted_rules_confSA_path, "w") as f:
#                 json.dump(data, f)
#         except:
#             try:
#                 with open(extracted_rules_confSA_path, "w") as f:
#                     json.dump(list(data), f)
#             except:
#                 print (name)
#     sys.stderr.write("All rules dumped!")
    
    if not extracted_file2data["ergRule2cnt"]: #####
        for name in extracted_file2data: 
            extracted_rules_confSA_path = os.path.join(extracted_rules_confSA_dir, name + ".json")
            with open(extracted_rules_confSA_path, "r") as f:
                extracted_file2data[name] = json.load(f) #####
    
    rule2comma_prob = {rule: rule2comma[rule] / (rule2comma[rule] + rule2noComma[rule]) for rule in rule2comma}
    commaProb_path = os.path.join(extracted_rules_confSA_dir, "rule2commaProb" + ".json") ####
    with open(commaProb_path, "w") as f:
        json.dump(rule2comma_prob, f)
#     sys.stderr.write('Start estimating PCFG from rule file!\n')
#     dgtrs2unaryRule2logProb, dgtrs2binaryRule2logProb = estimate_pcfg(extracted_file2data) # dmrsNodeSubgr2preTerm
#     sys.stderr.write('Done!\n')
#     sys.stderr.write('Start writing PCFG to file!\n')

    unary_pcfg_path = os.path.join(extracted_rules_confSA_dir, "dgtrs2unaryRule2logProb" + ".json") ####
    binary_pcfg_path = os.path.join(extracted_rules_confSA_dir, "dgtrs2binaryRule2logProb" + ".json") ###
    
    with open(unary_pcfg_path, "r") as f: ####
        dgtrs2unaryRule2logProb = json.load(f)
    with open(binary_pcfg_path, "r") as f:
        dgtrs2binaryRule2logProb = json.load(f) ####
        
    sys.stderr.write("PCFG log prob loaded!\n") #####
    dgtrsTrgs2edges2surf2logProb =\
        get_semEmtsubtree_logProb(extracted_file2data["dgtrsEdgesTrg2SubtreeDgtrs"],
                              dgtrs2unaryRule2logProb, dgtrs2binaryRule2logProb) #####
    sys.stderr.write("Augmented copula/semEmt dicts with logProb!\n") #####
    
#     dgtrsTrgs2edges2surf2logProb_copula, dgtrsTrgs2edges2surf2logProb_prtcl\
#         = [dgtrsTrgs2edges2surf2logProb[semEmt] for semEmt in ['copula', 'prtcl']]
    
    aug_copula_path = os.path.join(extracted_rules_confSA_dir, "dgtrsTrgs2edges2surf2logProb_copula" + ".json") ####
    aug_prtcl_path = os.path.join(extracted_rules_confSA_dir, "dgtrsTrgs2edges2surf2logProb_prtcl" + ".json") ###
    aug_semEmt_path = os.path.join(extracted_rules_confSA_dir, "dgtrsTrgs2edges2surf2logProb" + ".json") ### 
    
    with open(aug_semEmt_path, "w") as f: ####
        json.dump(dgtrsTrgs2edges2surf2logProb, f)
#     with open(aug_copula_path, "w") as f: ####
#         json.dump(dgtrsTrgs2edges2surf2logProb_copula, f)
#     with open(aug_prtcl_path, "w") as f: ####
#         json.dump(dgtrsTrgs2edges2surf2logProb_prtcl, f)
    sys.stderr.write("Dumped!\n") #####
    
    
#     sys.stderr.write('Extracting semi-canonical form of connected equal anchored DMRS subgraphs\n')
#     preTermCanon2surface2cnt = extracted_file2data["preTermCanon2surface2cnt"]
#     ccontCanon2intSubgrEdges2cnt = extracted_file2data["ccontCanon2intSubgrEdges2cnt"]
#     eqAnc_semiCanons = get_equalAnc_semiCanons(preTermCanon2surface2cnt).union(get_equalAnc_semiCanons(dmrsSubgrCanon2bTag2cnt)) 
#     eqAnc_semiCanons_usp = get_equalAnc_semiCanons(preTermCanonUsps2cnt)
#     ccont_semiCanons = get_equalAnc_semiCanons(ccontCanon2intSubgrEdges2cnt, keeplen1 = True)
#     sys.stderr.write('Done\n')
#     sys.stderr.write('Writing semi-canonical forms to files\n')
#     eqAnc_semiCanons_path = os.path.join(extracted_rules_confSA_dir, "eqAnc_semiCanons" + ".json")
#     eqAnc_semiCanons_usp_path = os.path.join(extracted_rules_confSA_dir, "eqAnc_semiCanons_usp" + ".json")
#     ccont_semiCanons_path = os.path.join(extracted_rules_confSA_dir, "ccont_semiCanons" + ".json")
#     with open(eqAnc_semiCanons_path, "w") as f: 
#         json.dump(list(eqAnc_semiCanons), f)
#     with open(eqAnc_semiCanons_usp_path, "w") as f:
#         json.dump(list(eqAnc_semiCanons_usp), f)
#     with open(ccont_semiCanons_path, "w") as f:
#         json.dump(list(ccont_semiCanons), f)
#     sys.stderr.write('Done\n')
    
    return semEmtLexEnt
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotatedData_dir', help='path to preprocessed data directory')
    parser.add_argument('extracted_rules_dir', help='path to extracted rules directory')
    parser.add_argument('redwdOrGw', help='redwoods or gigaword or combined')
    parser.add_argument('config_name', help='standard, dense, ...')
    parser.add_argument('sampleOrAll', help='try with sample data only or not')
    parser.add_argument('nodeTyped', help='pshrg rules are node-typed')
    parser.add_argument('stopAfter', help='Stop training after seeing the n-th gigaword instance')
    parser.add_argument('wsjOnly', help='train with wsj data only')
     
    args = parser.parse_args()
    main(args.annotatedData_dir, args.extracted_rules_dir, args.redwdOrGw, args.config_name, args.sampleOrAll, args.nodeTyped, args.stopAfter, args.wsjOnly)
    # main(args.annotatedData_dir, args.extracted_rules_dir, args.config_filepath)
