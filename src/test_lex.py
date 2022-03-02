# import delphin
# from delphin.codecs import simplemrs, dmrsjson
# from delphin import itsdb, util
from operator import add
import copy
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
from collections import deque
from ast import literal_eval as make_tuple
# from itertools import groupby
from itertools import chain
try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm
    
import networkx as nx
from networkx.algorithms.components import is_weakly_connected
from networkx.algorithms.boundary import edge_boundary 
import sacrebleu
    
# import pyximport; pyximport.install()
import src.util as util    
import src.eval_derivSim as eval_derivSim
import src.train_lex as train_lex
# import src.chart_parse_cy as chartParse

from src import dg_util, timeout
from networkx.drawing.nx_agraph import to_agraph


try:
    from pygraphviz import AGraph
except:
    pass

import time
import signal

timeoutSeconds = 300
forceSemEmt = False
no_semEmt = False
usePHRG = False
node_typed = False
hookCheck = False
lexicalized = False
max_subgraph_size = 6 # default as 6
min_preTermCanon_freq = 0
max_unary_chain = 3
indexFailMax = 0
commaProb_min = 0.6
startSymRstr = True

no_hyp = 0

rule2commaProb = defaultdict()
startRule2cnt = Counter()
ergRule2cnt = Counter()
dmrsSubgrCanon2bTag2cnt = defaultdict(Counter)
preTermCanon2bTagSurface2cnt = defaultdict(Counter) # has same keys as dmrsSubgrCanon2bTag2cnt
preTerm2surface2cnt = defaultdict(Counter)
preTermAnno2surface2cnt = defaultdict(Counter)
preTermCanon2surface2cnt = defaultdict(Counter)
preTermCanonUsps2cnt = Counter()
eqAnc_semiCanons = set()
eqAnc_semiCanons_usp = set()
ccont_semiCanons = set()
dgtrs2unaryRule2cnt = defaultdict(Counter)
dgtrs2unaryRule2logProb = defaultdict(Counter)
dgtrs2binaryRule2cnt = defaultdict(Counter)
dgtrs2binaryRule2logProb = defaultdict(Counter)
# SHRG In the format of (concerned_nodes_repl, concerned_edges_repl): derivRule
# e.g. ((ccont_pred, erg_Lrule, erg_Rrule), (CL_edge_lbls, CR_edge_lbls, LR_edge_lbls)): erg_BinCcontRule
# e.g. ((erg_Urule), ()): erg_Crule
SHRG = defaultdict(Counter)
SHRG_coarse = defaultdict(Counter)
PHRG = defaultdict(defaultdict)
ccontCanon2intSubgrEdges2cnt = defaultdict(Counter)
canon2cnt = Counter()
canon_usp2cnt = Counter()
dgtrsTrgs2edges2surf2logProb = defaultdict(lambda: defaultdict(lambda: defaultdict(defaultdict)))
predTrg2edge2surf2logProb = defaultdict(lambda: defaultdict(lambda: defaultdict(defaultdict)))
exactCanonsToMatch = set()
uspCanonsToMatch = set()

filtered_preTermCanonUsp = set()
ccont_preds = set()
SHRG_rules2edges = defaultdict(list)
SHRG_unaryTC = defaultdict(Counter)
dgtrs2unaryRuleTC2logProb = defaultdict(defaultdict)
dgtrPar2unaryChain = defaultdict(list)
semEmtDgtrsRule = defaultdict(set)
pred2dgtrs = defaultdict(set)
SRule2logProb = None
merge_hist_set = set()
snt_succ = False
total_succ_cnt = 0
total_Ssucc_cnt = 0
times_spent = []
bleus_list = []
processed_cnt = 0
unrealized2cnt = Counter()
sntId2log = defaultdict(defaultdict)

results_dir_suffix = "26012022"

def filter_preTermCanonUsp(canon_usp_cnt, min_preTermCanon_freq):
    global filtered_preTermCanonUsp
    
    for ki, (k, v) in enumerate(get_priorOfCanon(canon_usp2cnt).items()):
#         print (k,v,canon_usp2cnt[k])
        if canon_usp2cnt[k] < min_preTermCanon_freq: 
            continue
        filtered_preTermCanonUsp.add(k)

def get_priorOfCanon(canon2cnt):
    denom = sum(canon2cnt.values())
    canon2prior = {k: v/denom for k,v in canon2cnt.items()}
    return dict(sorted(canon2prior.items(), key=lambda x:x[1], reverse=True))

def get_intSubgrEdgesProp(ccontCanon2intSubgrEdges2cnt):
    pass
    
def get_extractedFileName2data():
    global rule2commaProb, startRule2cnt, ergRule2cnt, dmrsSubgrCanon2bTag2cnt, preTermCanon2bTagSurface2cnt, preTerm2surface2cnt, preTermAnno2surface2cnt, preTermCanon2surface2cnt, dgtrs2unaryRule2cnt, dgtrs2binaryRule2cnt, PHRG, SHRG, SHRG_coarse, ccontCanon2intSubgrEdges2cnt, dgtrsEdgesTrg2SubtreeDgtrs_semEmt, canon2cnt, canon_usp2cnt, dgtrs2unaryRule2logProb, dgtrs2binaryRule2logProb, eqAnc_semiCanons, eqAnc_semiCanons_usp, ccont_semiCanons
    extracted_file2data = {
                    'rule2commaProb': rule2commaProb,
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
                     "PHRG": PHRG,
                     "SHRG": SHRG,
                     "SHRG_coarse": SHRG_coarse,
                     "ccontCanon2intSubgrEdges2cnt": ccontCanon2intSubgrEdges2cnt,
                     "dgtrsTrgs2edges2surf2logProb": dgtrsTrgs2edges2surf2logProb,
                     "canon2cnt": canon2cnt,
                     "canon_usp2cnt": canon_usp2cnt,
                     "dgtrs2unaryRule2logProb": dgtrs2unaryRule2logProb,
                     "dgtrs2binaryRule2logProb": dgtrs2binaryRule2logProb,
                     "eqAnc_semiCanons": eqAnc_semiCanons,
                     "eqAnc_semiCanons_usp": eqAnc_semiCanons_usp,
                     "ccont_semiCanons": ccont_semiCanons}
    return extracted_file2data

def gen_hypothesis3(bitVec2rule2oldItems, oldItem2histLogP, dmrs_nxDG_ext, psvNodesBitVec2nbrsBitVec, bitVec2newPsvItems, 
                    bitVec2ccontCanon, edgeKey2lbl, pred2edgesBV2key2lbl, dmrsNode2scope, dmrs_node2bitVec, bitVec_complete, final_ltop, neg_edges, lexCanon2usp, node2pos
, node_typed):
    completed = defaultdict()
    Scompleted = defaultdict()
    semEmt_added_BVs = set()
    one, one_suc, two, two_suc, twofive, twofive_suc, three, three_suc, cop, cop_suc, sem, sem_suc = (0,0,0,0,0,0,0,0,0,0,0,0)
    # iterate for #nodes and end
    itr = -1
#     print (bitVec2rule2oldItems2histLogP)
#     check = [(1023, "sb-hd_mc_c"),
# (7, "sp-hd_n_c"),
# (6, "aj-hdn_norm_c"),
# (1016, "hd-cmp_u_c"),
# (1008, "hdn_bnp_c"),
# (1008, "aj-hdn_adjn_c"),
# (992, "aj-hdn_norm_c"),
# (960, "hdn-aj_redrel_c"),
# (896, "hd-cmp_u_c"),
# (768, "hdn_bnp_c"),
# (768, "aj-hdn_norm_c")]

#    check = [(63, "sb-hd_mc_c"),
#(3, "sp-hd_n_c"),
#(60, "sp-hd_hc_c"),
#(28, "mnp_deg-prd_c"),
#(28, "num-n_mnp_c")]
    check = [(2048 + 4096 + 8192 + 16384 + 32768 + 65536 + 131072, "hd_xaj-int-vp_c"),
             (1024 + 2048 + 4096 + 8192 + 16384 + 32768 + 65536 + 131072, "aj-hd_scp-xp_c")]
    check_bin = [4096 + 8192 + 16384 + 32768 + 65536 + 131072 + 262144, 2048 + 4096 + 8192 + 16384 + 32768 + 65536 + 131072 + 262144, 16 + 32 + 64]
    checked = set()
#     print (bin(bitVec_complete))
    while(1):
        _ = None
        itr += 1
        if not bitVec2newPsvItems:
#             pprint (checked)
            break
#         completed_psvNodesPair = set()
        checked_free = set()
        targ_bitVec = bitVec_complete - 3
        #unary
        # for max unary chain number of times, find out unary composition
        # only apply to dmrs nodes that are not applied before
        no_items = 1
        for bitVec, newPsvItems in bitVec2newPsvItems.items():          
            for (psv_derivRule, extEdges), (extEdges_dicts, semEmts_trgs, scope, indexFail) in newPsvItems.copy().items():
#                 print (psv_derivRule, semEmts_trgs)
                psv_derivRule_tmp = lexCanon2usp.get(psv_derivRule) or psv_derivRule
                copula_trg, prtcl_trg, compl_trgs = semEmts_trgs
                one += 1
                mode = 'U'
                shrgKey2psvItem = util.semCompMode2func[mode](dmrs_nxDG_ext, _,(psv_derivRule,), (psv_derivRule_tmp,), None, 
                                                            (extEdges,), (extEdges_dicts,), (semEmts_trgs,), (scope,), (bitVec,), 
                                                            SHRG_unaryTC, edgeKey2lbl, dmrs_node2bitVec,
                                                            no_items, mode = mode, node2pos = node2pos
, node_typed = node_typed)
                
                if shrgKey2psvItem:
                    one_suc += 1
                for shrgKey in shrgKey2psvItem:
                    for derivRule_new in SHRG_unaryTC[shrgKey]:
#                         if usePHRG:
#                             log_prob_new = PHRG[shrgKey][derivRule_new]
#                         else:
                        log_prob_new = dgtrs2unaryRuleTC2logProb[psv_derivRule_tmp][derivRule_new]
                        cum_log_prob_new = oldItem2histLogP[(bitVec, psv_derivRule, extEdges)][1]\
                            + log_prob_new
                        new_psv_item = (bitVec, derivRule_new, extEdges)
                        # local ambiguity factoring
                        if new_psv_item in oldItem2histLogP:
                            old_logProb = oldItem2histLogP[new_psv_item][1]
                            if old_logProb and cum_log_prob_new < old_logProb:
                                continue
                        bitVec2rule2oldItems[bitVec][derivRule_new].add(extEdges)
                        if len(dgtrPar2unaryChain[(psv_derivRule_tmp, derivRule_new)]) > 2:
                            oldItem2histLogP[new_psv_item] = (((bitVec,
                                                            ("UC", psv_derivRule, *dgtrPar2unaryChain[(psv_derivRule_tmp, derivRule_new)][1:-1]),
                                                            extEdges),),
                                                                             cum_log_prob_new, None, extEdges_dicts,
                                                                             semEmts_trgs, scope, indexFail) 
                        else:
                            
                            oldItem2histLogP[new_psv_item] = (((bitVec, psv_derivRule, extEdges),),
                                                                             cum_log_prob_new, None, extEdges_dicts,
                                                                             semEmts_trgs, scope, indexFail) 
                            
                        bitVec2newPsvItems[bitVec][(derivRule_new, extEdges)] = (extEdges_dicts, semEmts_trgs, scope, indexFail)
                        
        for bv in bitVec2newPsvItems:
            # check if coverage is full and new deriv rule in S
#             print (bv, bitVec_complete)for item in bitVec2newPsvItems[bv]:
            for item in bitVec2newPsvItems[bv]:
                for c in check:
                    if bv == c[0] and isinstance(item[0], str) and item[0].startswith(c[1]):
                        checked.add((bv, item[0], oldItem2histLogP[(bv, *item)][1]))
                # if bv in [65536, 2**16+2**15,2**16+2**15+2**14,2**16+2**15+2**14+2**13,2**16+2**15+2**14+2**13+2**12]:
                    # checked_free.add((bv, item[0], oldItem2histLogP[(bv, *item)][1]))
            if bv != bitVec_complete:
                continue
            
            for item in bitVec2newPsvItems[bv]:
                if item[0] not in SRule2logProb or not startSymRstr:
                    completed[(bv,*item)] = oldItem2histLogP[(bv, *item)]
                else:
                    final_histLogP = list(oldItem2histLogP[(bv, *item)])
                    if final_histLogP[1] == 0 or True:
                        final_histLogP[1] += SRule2logProb[item[0]]
                    Scompleted[(bv,*item)] = tuple(final_histLogP)
        
#         pprint (bitVec2newPsvItems)
#         print ([bin(bv) for bv in list(bitVec2newPsvItems.keys())])
#         for bv in bitVec2newPsvItems:
#             if bv == 4193792:s
#                 print(bv, bitVec2newPsvItems[bv])
#             print (bv, bin(bv))
#             for pi in bitVec2newPsvItems[bv]:
#                 if bv in [8160 - 8191 + 262143, # 262112
#                           8188 - 8191 + 262143, # 262140
#                           8184 - 8191 + 262143, # 262136
#                           4064,
#                           - 8191 + 262143 # 253952
#                          ]:
#                 print ("\t", pi[0])
#                 input()

        # pprint (checked_free)
    
        bitVec2newPsvItems_tmp = defaultdict(defaultdict)
        bitVec2rule2oldItems_tmp = bitVec2rule2oldItems.copy()
        oldItem2histLogP_tmp = oldItem2histLogP.copy()
        psvNodesBitVec2nbrsBitVec_tmp = psvNodesBitVec2nbrsBitVec.copy()
        
        # copula semEmt for binary (with no semEmt dgtr) with ccont pred
        no_items = 3
        if not noSemEmt:
            for idx, semEmt_type in enumerate(['copula', 'prtcl', 'compl', 'by']):
                for bitVec_ccont, (ccont_canon, ccont_extEdges, ccont_extEdges_dicts,
                                   ccont_semEmtTrgs, ccont_scope, ccont_indexFail) in bitVec2ccontCanon.items():
                    for bitVec1 in bitVec2newPsvItems:
                        if bitVec1 in bitVec2ccontCanon:
                            continue
                        for bitVec2 in bitVec2rule2oldItems:
                            if bitVec2 in bitVec2ccontCanon:
                                continue
                            if bitVec1 & bitVec2 != 0 or bitVec1 & bitVec_ccont or bitVec2 & bitVec_ccont != 0:
                                continue
                            if sum([1 if a & b != 0 else 0
                                    for a,b in [[psvNodesBitVec2nbrsBitVec[bitVec_ccont], bitVec1],
                                                [psvNodesBitVec2nbrsBitVec[bitVec1], bitVec2],
                                                [psvNodesBitVec2nbrsBitVec[bitVec2], bitVec_ccont]]]) < 2:
                                continue
                            has_rule = False
                            bitVec_new = bitVec2 | bitVec_ccont | bitVec1
                            for (psv_derivRule1, extEdges1), data1 in bitVec2newPsvItems[bitVec1].items():
        #                         print (psv_derivRule1, data1)
                                psv_derivRule1_tmp = lexCanon2usp.get(psv_derivRule1) or psv_derivRule1
                                extEdges_dicts1, semEmt_trgs1, scope1, indexFail1 = data1
                                for psv_derivRule2 in bitVec2rule2oldItems[bitVec2]:
                                    psv_derivRule2_tmp = lexCanon2usp.get(psv_derivRule2) or psv_derivRule2
                                    lOrR0 = 0
                                    if (ccont_canon, psv_derivRule1_tmp, psv_derivRule2_tmp) in semEmtDgtrsRule[semEmt_type]:
                                        lOrR0 += 1
                                    if (ccont_canon, psv_derivRule2_tmp, psv_derivRule1_tmp) in semEmtDgtrsRule[semEmt_type]:
                                        lOrR0 += 2
                                    if lOrR0 == 0:
                                        continue
                                    for extEdges2 in bitVec2rule2oldItems[bitVec2][psv_derivRule2]:
                                        if bitVec2 in bitVec2newPsvItems and (psv_derivRule2, extEdges2) in bitVec2newPsvItems[bitVec2]\
                                            and bitVec1 > bitVec2:
                                            continue
                                        extEdges_dicts2, semEmt_trgs2, scope2, indexFail2 =\
                                            oldItem2histLogP_tmp[(bitVec2, psv_derivRule2, extEdges2)][3:]
                                        semEmt_idx = idx
                                        if semEmt_type == 'by': semEmt_idx = 0
                                        ccont_semEmtTrg, semEmt_trg1, semEmt_trg2 = ccont_semEmtTrgs[semEmt_idx], semEmt_trgs1[semEmt_idx],\
                                                                                    semEmt_trgs2[semEmt_idx]
                                        lOrR = [False, False, False, False]
                                        if semEmt_trg1:
                                            if lOrR0 % 2 == 1:
                                                if ((ccont_canon, None), (psv_derivRule1_tmp, semEmt_trg1), (psv_derivRule2_tmp, None))\
                                                    in dgtrsTrgs2edges2surf2logProb[semEmt_type]: lOrR[0] = True
                                            if lOrR0 >= 2:
                                                if ((ccont_canon, None), (psv_derivRule2_tmp, None), (psv_derivRule1_tmp, semEmt_trg1))\
                                                    in dgtrsTrgs2edges2surf2logProb[semEmt_type]: lOrR[1] = True
                                        if semEmt_trg2:
                                            if lOrR0 % 2 == 1:
                                                if ((ccont_canon, None), (psv_derivRule1_tmp, None), (psv_derivRule2_tmp, semEmt_trg2))\
                                                    in dgtrsTrgs2edges2surf2logProb[semEmt_type]: lOrR[2] = True
                                            if lOrR0 >= 2:
                                                if ((ccont_canon, None), (psv_derivRule2_tmp, semEmt_trg2), (psv_derivRule1_tmp, None))\
                                                    in dgtrsTrgs2edges2surf2logProb[semEmt_type]: lOrR[3] = True
                                        if any(lOrR):
                                            p = False
                                            cop += 1
                                            derivRule2psvItem = util.matched_semComp_semEmt(dmrs_nxDG_ext,
                                                                        (ccont_canon, psv_derivRule1, psv_derivRule2), ((ccont_canon, None),
                                                                    (psv_derivRule1_tmp, semEmt_trg1), (psv_derivRule2_tmp, semEmt_trg2)), None,
                                                                      (ccont_extEdges, extEdges1, extEdges2),
                                                                    (ccont_extEdges_dicts, extEdges_dicts1, extEdges_dicts2), 
                                                                        (ccont_semEmtTrgs, semEmt_trgs1, semEmt_trgs2),
                                                                    (ccont_scope, scope1, scope2), (bitVec_ccont, bitVec1, bitVec2),
                                                                            dgtrsTrgs2edges2surf2logProb[semEmt_type],
                                                                    edgeKey2lbl, dmrs_node2bitVec, no_items, mode = semEmt_type, p = p,
                                                                                          directed = lOrR, return_exts = True,  node2pos = node2pos
, node_typed = node_typed)
                                            if derivRule2psvItem: cop_suc += 1
                                            for shrgKey, psvItems_new in derivRule2psvItem.items():
                                                derivRule_new, interSubgrsEdges = shrgKey
                                                for (new_bitVecs, new_dgtr_derivRules, new_semEmtsTrg, new_dgtr_scopes, new_logProb, surf,
                                                     new_extEdge, new_extEdges_dict, new_dgtr_extEdges, extEdgeKeyExts_list, interSubgrsEdges_keys) in psvItems_new:
                                                    full_itemC = (new_bitVecs[0], new_dgtr_derivRules[0], new_dgtr_extEdges[0])
                                                    full_itemL = (new_bitVecs[1], new_dgtr_derivRules[1], new_dgtr_extEdges[1])
                                                    full_itemR = (new_bitVecs[2], new_dgtr_derivRules[2], new_dgtr_extEdges[2])
                                                    new_scope = None
                                                    new_indexFailCnt = 0
                                                    if hookCheck:
                                                        # check index availability here
                                                        index_aval = util.index_aval_check(derivRule_new, extEdgeKeyExts_list,
                                                                                          edgeKey2lbl, no_items)
                                                        new_indexFailCnt = indexFail1 + indexFail2
                                                        if not index_aval:
                                                            if not any(["_inv_" in s for s in surf]):
                                                                contains_neg_edge = any([e in edge_set for e in neg_edges
                                                                                         for edge_set in interSubgrsEdges_keys])
                                                                if not contains_neg_edge:
                                                                    new_indexFailCnt += 1
                                                        if new_indexFailCnt > indexFailMax:
                                                            continue
                                                        ltop_aval, new_scope = util.ltop_aval_check(derivRule_new, extEdgeKeyExts_list, interSubgrsEdges,
                                                                                          edgeKey2lbl, new_dgtr_scopes, dmrsNode2scope, no_items)
                                                        if not ltop_aval and not (semEmt_type == 'copula'
                                                                              and ((semEmt_trg1 and "sf:Q" in semEmt_trg1)
                                                                                    or (semEmt_trg2 and "sf:Q" in semEmt_trg2))):
                                                            continue
                                                        if final_ltop in new_dgtr_scopes and new_scope and new_scope != final_ltop:
                                                            continue
                                                    if usePHRG:
                                                        log_prob_new = -0.01
                                                    else:
                                                        log_prob_new = new_logProb
                                                    cum_log_prob_new = oldItem2histLogP_tmp[full_itemL][1]\
                                                            + oldItem2histLogP_tmp[full_itemR][1] + log_prob_new
                                                    full_new_psv_item = (bitVec_new, derivRule_new, new_extEdge)
                                                    if full_new_psv_item in oldItem2histLogP_tmp:
                                                        old_logProb = oldItem2histLogP_tmp[full_new_psv_item][1]
                                                        if old_logProb and cum_log_prob_new < old_logProb:
                                                            continue
                                                    has_rule = True
                                                    if forceSemEmt and semEmt_type in ['copula', 'by']:
                                                        semEmt_added_BVs.add(new_bitVecs)
                                                    # print ("added", semEmt_type, surf)
                                                    # print ((psv_derivRule1, semEmt_trg1), (psv_derivRule2, semEmt_trg2))
                                                    # print ()
                    #                               # prop copula and semEmt (done in matched_semComp)
                                                    oldItem2histLogP_tmp[full_new_psv_item] = (
                                                        (full_itemC, full_itemL, full_itemR),
                                                         cum_log_prob_new, surf, new_extEdges_dict, new_semEmtsTrg, new_scope, new_indexFailCnt
                                                    )
                                                    bitVec2newPsvItems_tmp[bitVec_new][(derivRule_new, new_extEdge)] = new_extEdges_dict,\
                                                        new_semEmtsTrg, new_scope, new_indexFailCnt
                                                    bitVec2rule2oldItems_tmp[bitVec_new][derivRule_new].add(new_extEdge)

                            if has_rule:
                                if bitVec_new in psvNodesBitVec2nbrsBitVec_tmp:
                                    try:
                                        assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                            == (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]\
                                                   | psvNodesBitVec2nbrsBitVec[bitVec_ccont]) & ~bitVec_new
                                    except:
                                        print ("neighbour inconsistent")
                                        print (psvNodesBitVec2nbrsBitVec)
                                        print ((bitVec_new),(bitVec1), (bitVec2), (bitVec_ccont))
                                        print (bin(bitVec_new),bin(bitVec1), bin(bitVec2), bin(bitVec_ccont))
                                        # input()
                                else:
                                    psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                        = (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]\
                                              | psvNodesBitVec2nbrsBitVec[bitVec_ccont]) & ~bitVec_new
                                    assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new] >= 0


        
        #copula, sememt for binary
        no_items = 2
        if not noSemEmt:
            for idx, semEmt_type in enumerate(['copula', 'prtcl', 'compl', 'by']):
                for bitVec1 in bitVec2newPsvItems:
                    if bitVec1 in bitVec2ccontCanon: continue
                    for bitVec2 in bitVec2rule2oldItems:
                        if bitVec2 in bitVec2ccontCanon: continue
                        if bitVec1 == bitVec2 or bitVec1 & bitVec2 != 0: continue
                        if psvNodesBitVec2nbrsBitVec[bitVec1] & bitVec2 == 0:
                            assert psvNodesBitVec2nbrsBitVec[bitVec2] & bitVec1 == 0
                            continue
                        assert psvNodesBitVec2nbrsBitVec[bitVec2] & bitVec1 != 0
                        assert psvNodesBitVec2nbrsBitVec[bitVec1] & bitVec2 != 0
                        has_rule = False
                        bitVec_new = bitVec1 | bitVec2
    #                     if bitVec1 == 262112:
    #                         print (bitVec1, bitVec2newPsvItems)
                        for (psv_derivRule1, extEdges1), (extEdges_dicts1, semEmt_trgs1, scope1, indexFail1) in bitVec2newPsvItems[bitVec1].items():
                            psv_derivRule1_tmp = lexCanon2usp.get(psv_derivRule1) or psv_derivRule1
    #                         if bitVec1 == 262112 or "np-np-crd-t_c" in psv_derivRule1:
    #                             print (1, psv_derivRule1_tmp)
                            for psv_derivRule2 in bitVec2rule2oldItems[bitVec2]:
                                psv_derivRule2_tmp = lexCanon2usp.get(psv_derivRule2) or psv_derivRule2
                                lOrR0 = 0
                                if (psv_derivRule1_tmp, psv_derivRule2_tmp) in semEmtDgtrsRule[semEmt_type]:
                                    lOrR0 += 1
                                if (psv_derivRule2_tmp, psv_derivRule1_tmp) in semEmtDgtrsRule[semEmt_type]:
                                    lOrR0 += 2
    #                             if 'neg' in str(psv_derivRule2) and 'hd-cmp_u_c&VP' in str(psv_derivRule1):
    #                                 print (psv_derivRule1)
    #                                 print (lOrR0)
                                if lOrR0 == 0:
                                    continue
    #                             print (psv_derivRule1, psv_derivRule2)
                                for extEdges2 in bitVec2rule2oldItems[bitVec2][psv_derivRule2]:
                                    if bitVec2 in bitVec2newPsvItems and (psv_derivRule2, extEdges2) in bitVec2newPsvItems[bitVec2]\
                                        and bitVec1 > bitVec2:
                                        continue
                                    extEdges_dicts2, semEmt_trgs2, scope2, indexFail2 = oldItem2histLogP_tmp[(bitVec2, psv_derivRule2,
                                                                                                              extEdges2)][3:]
                                    semEmt_idx = idx
                                    if semEmt_type == 'by': semEmt_idx = 0
                                    semEmt_trg1, semEmt_trg2 = semEmt_trgs1[semEmt_idx], semEmt_trgs2[semEmt_idx]
                                    lOrR = [False, False, False, False]
                                    if semEmt_trg1:
                                        if lOrR0 % 2 == 1:
                                            if ((psv_derivRule1_tmp, semEmt_trg1), (psv_derivRule2_tmp, None))\
                                                in dgtrsTrgs2edges2surf2logProb[semEmt_type]: lOrR[0] = True
                                        if lOrR0 >= 2:
                                            if ((psv_derivRule2_tmp, None), (psv_derivRule1_tmp, semEmt_trg1))\
                                                in dgtrsTrgs2edges2surf2logProb[semEmt_type]: lOrR[1] = True
                                    if semEmt_trg2:
                                        if lOrR0 % 2 == 1:
                                            if ((psv_derivRule1_tmp, None), (psv_derivRule2_tmp, semEmt_trg2))\
                                                in dgtrsTrgs2edges2surf2logProb[semEmt_type]: lOrR[2] = True
                                        if lOrR0 >= 2:
                                            if ((psv_derivRule2_tmp, semEmt_trg2), (psv_derivRule1_tmp, None))\
                                                in dgtrsTrgs2edges2surf2logProb[semEmt_type]: lOrR[3] = True
                                    if any(lOrR):
                                        p = False
                                        cop += 1
    #                                     print (semEmt_trgs1, semEmt_trgs2, "orig!")
                                        derivRule2psvItem = util.matched_semComp_semEmt(dmrs_nxDG_ext,
                                                                                    (psv_derivRule1, psv_derivRule2),
                                                                ((psv_derivRule1_tmp, semEmt_trg1), (psv_derivRule2_tmp, semEmt_trg2)), None,
                                                                  (extEdges1, extEdges2),
                                                                (extEdges_dicts1, extEdges_dicts2), 
                                                                    (semEmt_trgs1, semEmt_trgs2),
                                                                (scope1, scope2), (bitVec1, bitVec2),
                                                                        dgtrsTrgs2edges2surf2logProb[semEmt_type],
                                                                edgeKey2lbl, dmrs_node2bitVec, no_items, mode = semEmt_type, p = p,
                                                                                      directed = lOrR, return_exts = True, node2pos = node2pos
, node_typed = node_typed)
                                        if derivRule2psvItem: cop_suc += 1
                                        for shrgKey, psvItems_new in derivRule2psvItem.items():
    #                                         pprint (psvItems_new)
                                            derivRule_new, interSubgrsEdges = shrgKey
                                            for (new_bitVecs, new_dgtr_derivRules, new_semEmtsTrg, new_dgtr_scopes, new_logProb, surf,
                                                 new_extEdge, new_extEdges_dict, new_dgtr_extEdges, extEdgeKeyExts_list,
                                                 interSubgrsEdges_keys) in psvItems_new:
                                                full_itemL = (new_bitVecs[0], new_dgtr_derivRules[0], new_dgtr_extEdges[0])
                                                full_itemR = (new_bitVecs[1], new_dgtr_derivRules[1], new_dgtr_extEdges[1])

                                                new_scope = None
                                                new_indexFailCnt = 0
                                                if hookCheck:
                                                    # check index availability here
                                                    index_aval = util.index_aval_check(derivRule_new, extEdgeKeyExts_list,
                                                                                      edgeKey2lbl, no_items)
                                                    new_indexFailCnt = indexFail1 + indexFail2
                                                    if not index_aval:
                                                        if not any(["_inv_" in s for s in surf]):
                                                            contains_neg_edge = any([e in edge_set for e in neg_edges
                                                                                     for edge_set in interSubgrsEdges_keys])
                                                            if not contains_neg_edge:
                                                                new_indexFailCnt += 1
                                                    if new_indexFailCnt > indexFailMax:
                                                        continue
                                                    ltop_aval, new_scope = util.ltop_aval_check(derivRule_new, extEdgeKeyExts_list, interSubgrsEdges,
                                                                                      edgeKey2lbl, new_dgtr_scopes, dmrsNode2scope, no_items)
                                                    if not ltop_aval and not (semEmt_type == 'copula'
                                                                              and ((semEmt_trg1 and "sf:Q" in semEmt_trg1)
                                                                                    or (semEmt_trg2 and "sf:Q" in semEmt_trg2))):
                                                        continue
                                                    if final_ltop in new_dgtr_scopes and new_scope and new_scope != final_ltop:
                                                        continue
                                                if usePHRG:
                                                    log_prob_new = -0.01
                                                else:
                                                    log_prob_new = new_logProb
                                                cum_log_prob_new = oldItem2histLogP_tmp[full_itemL][1]\
                                                        + oldItem2histLogP_tmp[full_itemR][1] + log_prob_new
                                                full_new_psv_item = (bitVec_new, derivRule_new, new_extEdge)
                                                if full_new_psv_item in oldItem2histLogP_tmp:
                                                    old_logProb = oldItem2histLogP_tmp[full_new_psv_item][1]
                                                    if old_logProb and cum_log_prob_new < old_logProb:
                                                        continue
                                                has_rule = True
                                                if forceSemEmt and semEmt_type in ['copula', 'by']:
                                                    semEmt_added_BVs.add(new_bitVecs)
                                       #          print ("added", semEmt_type, surf, lOrR)
                                       #          print ()
                                       #          print ((psv_derivRule1, semEmt_trg1), (psv_derivRule2, semEmt_trg2))
                                       #          #print ()
                                       #          #print (full_itemL)
                                       #          print ()
                                       #          print (derivRule_new)
                                       #          print ()
                                       #          print ()
                    #                           # prop copula and semEmt (done in matched_semComp)
                                                oldItem2histLogP_tmp[full_new_psv_item] = (
                                                    (full_itemL, full_itemR),
                                                     cum_log_prob_new, surf, new_extEdges_dict, new_semEmtsTrg, new_scope, new_indexFailCnt
                                                )
                                                bitVec2newPsvItems_tmp[bitVec_new][(derivRule_new, new_extEdge)] = new_extEdges_dict,\
                                                    new_semEmtsTrg, new_scope, new_indexFailCnt
                                                bitVec2rule2oldItems_tmp[bitVec_new][derivRule_new].add(new_extEdge)
                        if has_rule:
                            if bitVec_new in psvNodesBitVec2nbrsBitVec_tmp:
        #                         print (psvNodesBitVec2nbrsBitVec_tmp)
                                assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                == (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]) & ~bitVec_new
                            else:
                                psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                = (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]) & ~bitVec_new

                                assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new] >= 0
            
        #binary predSemEmt
        no_items = 2
        if not noSemEmt:
            for bitVec1 in bitVec2newPsvItems:
                if bitVec1 in bitVec2ccontCanon:
                    continue
                for bitVec2 in bitVec2rule2oldItems:
                    if bitVec2 in bitVec2ccontCanon:
                        continue
                    if bitVec1 == bitVec2 or bitVec1 & bitVec2 != 0:
                        continue
                    if psvNodesBitVec2nbrsBitVec[bitVec1] & bitVec2 == 0:
                        assert psvNodesBitVec2nbrsBitVec[bitVec2] & bitVec1 == 0
                        continue
                    assert psvNodesBitVec2nbrsBitVec[bitVec2] & bitVec1 != 0
                    assert psvNodesBitVec2nbrsBitVec[bitVec1] & bitVec2 != 0
                    bitVec_new = bitVec1 | bitVec2
                    for pred, dgtrs in pred2dgtrs.items():
    #                     print (pred2edgesBV2key2lbl)
    #                     input()
                        for edgeBV, key2lbl in pred2edgesBV2key2lbl[pred].items():
                            srcBV, targBV = edgeBV
    #                         print (bin(srcBV), bin(targBV))
                            if not (bitVec1 & srcBV and bitVec2 & targBV or bitVec1 & targBV and bitVec2 & srcBV):
                                continue
                            has_rule = False
                            for (psv_derivRule1, extEdges1), data1 in bitVec2newPsvItems[bitVec1].items():
                                psv_derivRule1_tmp = lexCanon2usp.get(psv_derivRule1) or psv_derivRule1
                                for psv_derivRule2 in bitVec2rule2oldItems[bitVec2]:
                                    psv_derivRule2_tmp = lexCanon2usp.get(psv_derivRule2) or psv_derivRule2
                                    lOrR_semEmt, directed_semEmt = 0, True
                                    if (psv_derivRule1_tmp, psv_derivRule2_tmp) in dgtrs:
                                        lOrR_semEmt += 1
                                    if (psv_derivRule2_tmp, psv_derivRule1_tmp) in dgtrs:
                                        lOrR_semEmt += 2
                                    if lOrR_semEmt == 0:
                                        continue
                                    if lOrR_semEmt == 3: directed_semEmt = False
                                    for extEdges2 in bitVec2rule2oldItems[bitVec2][psv_derivRule2]:
                                        if bitVec2 in bitVec2newPsvItems and (psv_derivRule2, extEdges2) in bitVec2newPsvItems[bitVec2]\
                                            and bitVec1 > bitVec2:
                                            continue
                                        extEdges_dicts1, semEmt_trgs1, scope1, indexFail1 = data1
                                        extEdges_dicts2, semEmt_trgs2, scope2, indexFail2 = oldItem2histLogP_tmp[
                                            (bitVec2, psv_derivRule2, extEdges2)][3:]
                                        p = False
                                        if lOrR_semEmt:
    #                                         print (psv_derivRule1, psv_derivRule2)
                                            mode = 'B-predSemEmt'
                                            if lOrR_semEmt == 2:
                                                shrgKey2psvItem = util.semCompMode2func[mode](dmrs_nxDG_ext, _,
                                                                                    (psv_derivRule2, psv_derivRule1),
                                                                                    (psv_derivRule2_tmp, psv_derivRule1_tmp), None,
                                                                                    (extEdges2, extEdges1),
                                                                                        (extEdges_dicts2, extEdges_dicts1),
                                                                                        (semEmt_trgs2, semEmt_trgs1), (scope2, scope1),
                                                                                        (bitVec2, bitVec1),
                                                                                        dgtrsTrgs2edges2surf2logProb['predSemEmt'],
                                                                                            edgeKey2lbl, dmrs_node2bitVec,
                                                                                no_items, mode = mode, p = p, directed = directed_semEmt,
                                                                                           return_exts = True,  node2pos = node2pos
, node_typed = node_typed)
                                            else:
                                                shrgKey2psvItem = util.semCompMode2func[mode](dmrs_nxDG_ext, _,
                                                                                    (psv_derivRule1, psv_derivRule2),
                                                                                    (psv_derivRule1_tmp, psv_derivRule2_tmp), None,
                                                                                    (extEdges1, extEdges2),
                                                                                        (extEdges_dicts1, extEdges_dicts2),
                                                                                        (semEmt_trgs1, semEmt_trgs2), (scope1, scope2),
                                                                                        (bitVec1, bitVec2),
                                                                                        dgtrsTrgs2edges2surf2logProb['predSemEmt'],
                                                                                            edgeKey2lbl, dmrs_node2bitVec,
                                                                                no_items, mode = mode, p = p, directed = directed_semEmt,
                                                                                        return_exts = True,  node2pos = node2pos
, node_typed = node_typed)
                                            two_suc += 1
                                            for shrgKey, (new_bitVecs, new_semEmtTrgs, new_dgtr_scopes, new_extEdge,
                                                          new_extEdges_dict, new_dgtr_extEdges, extEdgeKeyExts_list,
                                                          interSubgrsEdges_keys) in shrgKey2psvItem.items():
                                                (new_dgtr_derivRules, interSubgrsEdges), new_dgtr_derivRules_exact = shrgKey
                                                full_itemL = (new_bitVecs[0], new_dgtr_derivRules_exact[0], new_dgtr_extEdges[0])
                                                full_itemR = (new_bitVecs[1], new_dgtr_derivRules_exact[1], new_dgtr_extEdges[1])
                                                for (derivRule_new, surf, predTrg), new_logProb in dgtrsTrgs2edges2surf2logProb\
                                                    ['predSemEmt'][new_dgtr_derivRules][interSubgrsEdges].items():
                                                    # check predTrgEdge
                                                    is_edgeLbl_equal = False
                                                    _, edgeLbl, edgeSet_idx = predTrg
                                                    for edge_key, lbl in key2lbl.items():
                                                        if lbl == edgeLbl:
                                                            if any([edgeLbl in edge_ext for edge_ext in interSubgrsEdges[edgeSet_idx]]):
                                                                is_edgeLbl_equal = True
                                                    if not is_edgeLbl_equal: continue

                                                    new_scope = None
                                                    new_indexFailCnt = 0
                                                    if hookCheck:
                                                        # check index availability here
                                                        index_aval = util.index_aval_check(derivRule_new, extEdgeKeyExts_list,
                                                                                          edgeKey2lbl, no_items)
                                                        new_indexFailCnt = indexFail1 + indexFail2
                                                        if not index_aval:
                                                            contains_neg_edge = any([e in edge_set for e in neg_edges
                                                                                     for edge_set in interSubgrsEdges_keys])
                                                            if not contains_neg_edge:
                                                                new_indexFailCnt += 1
                                                        if new_indexFailCnt > indexFailMax:
                                                            continue
                                                        ltop_aval, new_scope = util.ltop_aval_check(derivRule_new, extEdgeKeyExts_list,
                                                                                                    interSubgrsEdges, edgeKey2lbl,
                                                                                                    new_dgtr_scopes, dmrsNode2scope, no_items, p = p)
                                                        if not ltop_aval:
                                                            continue
                                                        if final_ltop in new_dgtr_scopes and new_scope and new_scope != final_ltop:
                                                            continue
                                                    if usePHRG:
                                                        log_prob_new = -0.01
                                                    else:
                                                        log_prob_new = new_logProb
                                                    cum_log_prob_new = oldItem2histLogP_tmp[full_itemL][1]\
                                                        + oldItem2histLogP_tmp[full_itemR][1] + log_prob_new
                                                    full_new_psv_item = (bitVec_new, derivRule_new, new_extEdge)
                                                    if full_new_psv_item in oldItem2histLogP_tmp:
                                                        old_logProb = oldItem2histLogP_tmp[full_new_psv_item][1]
                                                        if old_logProb and cum_log_prob_new < old_logProb:
                                                            continue
                                                        # do nth if already handled by composition with semEmts above
                                                        if oldItem2histLogP_tmp[full_new_psv_item][2]:
                                                            if (full_itemL, full_itemR) == oldItem2histLogP_tmp[full_new_psv_item][0]\
                                                                and cum_log_prob_new < oldItem2histLogP_tmp[full_new_psv_item][1]:
                                                                continue
                                                    has_rule = True
                                                    if forceSemEmt:
                                                        semEmt_added_BVs.add((new_bitVecs[0], new_bitVecs[1]))
                    #                               # prop copula and semEmt
                                                    trgs_new = util.propagate_semEmtInfo_test(derivRule_new, new_semEmtTrgs)
                                                    oldItem2histLogP_tmp[full_new_psv_item] = (
                                                        (full_itemL, full_itemR),
                                                         cum_log_prob_new, surf, new_extEdges_dict, trgs_new, new_scope, new_indexFailCnt
                                                    )
                                                    bitVec2newPsvItems_tmp[bitVec_new][(derivRule_new, new_extEdge)] = new_extEdges_dict,\
                                                        trgs_new, new_scope, new_indexFailCnt
                                                    bitVec2rule2oldItems_tmp[bitVec_new][derivRule_new].add(new_extEdge)
                            if has_rule:
                                if bitVec_new in psvNodesBitVec2nbrsBitVec_tmp:
            #                         print (psvNodesBitVec2nbrsBitVec_tmp)
                                    assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                    == (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]) & ~bitVec_new
                                else:
                                    psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                    = (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]) & ~bitVec_new

                                    assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new] >= 0
                            
        # binary (with no semEmt dgtr) with ccont pred (predSemEmt)
        no_items = 3
        if not noSemEmt:
            for bitVec_ccont, (ccont_canon, ccont_extEdges, ccont_extEdges_dicts,
                               ccont_semEmtTrgs, ccont_scope, ccont_indexFail) in bitVec2ccontCanon.items():
                for bitVec1 in bitVec2newPsvItems:
                    if bitVec1 in bitVec2ccontCanon:
                        continue
                    for bitVec2 in bitVec2rule2oldItems:
                        if bitVec2 in bitVec2ccontCanon:
                            continue
                        if bitVec1 & bitVec2 != 0 or bitVec1 & bitVec_ccont or bitVec2 & bitVec_ccont != 0:
                            continue
                        if sum([1 if a & b != 0 else 0
                                for a,b in [[psvNodesBitVec2nbrsBitVec[bitVec_ccont], bitVec1],
                                            [psvNodesBitVec2nbrsBitVec[bitVec1], bitVec2],
                                            [psvNodesBitVec2nbrsBitVec[bitVec2], bitVec_ccont]]]) < 2:
                            continue
                        bitVec_new = bitVec_ccont | bitVec1 | bitVec2
                        has_rule = False
                        for pred, dgtrs in pred2dgtrs.items():
        #                     print (pred2edgesBV2key2lbl)
        #                     input()
                            for edgeBV, key2lbl in pred2edgesBV2key2lbl[pred].items():
                                srcBV, targBV = edgeBV
        #                         print (bin(srcBV), bin(targBV))
                                if (bitVec1 & srcBV and bitVec1 & targBV) or (bitVec2 & srcBV and bitVec2 & targBV)\
                                    or (bitVec_ccont & srcBV and bitVec_ccont & targBV) or (not srcBV & bitVec_new) or (not targBV & bitVec_new):
                                    continue
                                for (psv_derivRule1, extEdges1), data1 in bitVec2newPsvItems[bitVec1].items():
                                    psv_derivRule1_tmp = lexCanon2usp.get(psv_derivRule1) or psv_derivRule1
                                    extEdges_dicts1, semEmt_trgs1, scope1, indexFail1 = data1
                                    for psv_derivRule2 in bitVec2rule2oldItems[bitVec2]:
                                        psv_derivRule2_tmp = lexCanon2usp.get(psv_derivRule2) or psv_derivRule2
                                        lOrR_semEmt, directed_semEmt = 0, True
                                        if (ccont_canon, psv_derivRule1_tmp, psv_derivRule2_tmp) in dgtrs:
                                            lOrR_semEmt += 1
                                        if (ccont_canon, psv_derivRule2_tmp, psv_derivRule1_tmp) in dgtrs:
                                            lOrR_semEmt += 2
                                        if lOrR_semEmt == 0:
                                            continue
                                        if lOrR_semEmt == 3: directed_semEmt = False
                                        for extEdges2 in bitVec2rule2oldItems[bitVec2][psv_derivRule2]:
                                            if bitVec2 in bitVec2newPsvItems and (psv_derivRule2, extEdges2) in bitVec2newPsvItems[bitVec2]\
                                                and bitVec1 > bitVec2:
                                                continue
                                            extEdges_dicts2, semEmt_trgs2, scope2, indexFail2 = oldItem2histLogP_tmp[(bitVec2,
                                                                                                                psv_derivRule2, extEdges2)][3:]
                                            new_indexFailCnt = indexFail1 + indexFail2

                                            p = False

                                            if "sp-hd" in psv_derivRule1 and "comp" in str(psv_derivRule2):
    #                                             print (ccont_canon, psv_derivRule2, psv_derivRule1)
    #                                             print (dgtrs)
    #                                             print (lOrR_semEmt)
                                                p = True
                                            three += 1
                                            mode = 'BP-predSemEmt'
                                            if lOrR_semEmt == 2:
                                                shrgKey2psvItem = util.semCompMode2func[mode](dmrs_nxDG_ext, _,
                                                                                    (ccont_canon, psv_derivRule2, psv_derivRule1),
                                                                                    (ccont_canon, psv_derivRule2_tmp, psv_derivRule1_tmp), None,
                                                                                    (ccont_extEdges, extEdges2, extEdges1),
                                                                                    (ccont_extEdges_dicts, extEdges_dicts2, extEdges_dicts1),
                                                                                        (ccont_semEmtTrgs, semEmt_trgs2, semEmt_trgs2),
                                                                                        (ccont_scope, scope2, scope1),
                                                                                        (bitVec_ccont, bitVec2, bitVec1),
                                                                                            dgtrsTrgs2edges2surf2logProb['predSemEmt'],
                                                                                            edgeKey2lbl, dmrs_node2bitVec,
                                                                            no_items, mode = mode, p = p, directed = directed_semEmt,
                                                                                           return_exts = True,  node2pos = node2pos
, node_typed = node_typed)
                                            else:
                                                shrgKey2psvItem = util.semCompMode2func[mode](dmrs_nxDG_ext, _, 
                                                                                    (ccont_canon, psv_derivRule1, psv_derivRule2),
                                                                                    (ccont_canon, psv_derivRule1_tmp, psv_derivRule2_tmp), None,
                                                                                    (ccont_extEdges, extEdges1, extEdges2),
                                                                                    (ccont_extEdges_dicts, extEdges_dicts1, extEdges_dicts2),
                                                                                        (ccont_semEmtTrgs, semEmt_trgs1, semEmt_trgs2),
                                                                                        (ccont_scope, scope1, scope2),
                                                                                        (bitVec_ccont, bitVec1, bitVec2),
                                                                                            dgtrsTrgs2edges2surf2logProb['predSemEmt'],
                                                                                            edgeKey2lbl, dmrs_node2bitVec,
                                                                             no_items, mode = mode, p = p, directed = directed_semEmt,
                                                                                           return_exts = True,  node2pos = node2pos
, node_typed = node_typed)
                                            if not shrgKey2psvItem:
                                                continue
                                            three_suc += 1
                                            for shrgKey, (new_bitVecs, new_semEmtTrgs, new_dgtr_scopes, new_extEdge,
                                                                  new_extEdges_dict, new_dgtr_extEdges, extEdgeKeyExts_list,
                                                                  interSubgrsEdges_keys) in shrgKey2psvItem.items():
                                                (new_dgtr_derivRules, interSubgrsEdges), new_dgtr_derivRules_exact = shrgKey
                                                full_itemC = (new_bitVecs[0], new_dgtr_derivRules_exact[0], new_dgtr_extEdges[0])
                                                full_itemL = (new_bitVecs[1], new_dgtr_derivRules_exact[1], new_dgtr_extEdges[1])
                                                full_itemR = (new_bitVecs[2], new_dgtr_derivRules_exact[2], new_dgtr_extEdges[2])
                                                for (derivRule_new, surf, predTrg), new_logProb in dgtrsTrgs2edges2surf2logProb\
                                                    ['predSemEmt'][new_dgtr_derivRules][interSubgrsEdges].items():
                                                    # check predTrgEdge
                                                    is_edgeLbl_equal = False
                                                    _, edgeLbl, edgeSet_idx = predTrg
                                                    for edge_key, lbl in key2lbl.items():
                                                        if lbl == edgeLbl:
                                                            if any([edgeLbl in edge_ext for edge_ext in interSubgrsEdges[edgeSet_idx]]):
                                                                is_edgeLbl_equal = True
                                                    if not is_edgeLbl_equal: continue
                                                    new_scope = None
                                                    new_indexFailCnt = 0
                                                    if hookCheck:
                                                        index_aval = util.index_aval_check(derivRule_new, extEdgeKeyExts_list,
                                                                                          edgeKey2lbl, no_items)
                                                        if not index_aval:
                                                            contains_neg_edge = any([e in edge_set for e in neg_edges
                                                                                     for edge_set in interSubgrsEdges_keys])
                                                            if not contains_neg_edge:
                                                                new_indexFailCnt += 1
                                                        if new_indexFailCnt > indexFailMax:
                                                            continue
                                                        ltop_aval, new_scope = util.ltop_aval_check(derivRule_new, extEdgeKeyExts_list, interSubgrsEdges,
                                                                                              edgeKey2lbl, new_dgtr_scopes, dmrsNode2scope, 
                                                                                                    no_items, p)
                                                        if not ltop_aval:
                                                            continue
                                                        if final_ltop in new_dgtr_scopes and new_scope and new_scope != final_ltop:
                                                            continue
                                                    if usePHRG:
                                                        log_prob_new = -0.01
                                                    else:
                                                        log_prob_new = new_logProb
                                                    cum_log_prob_new = oldItem2histLogP_tmp[full_itemL][1]\
                                                        + oldItem2histLogP_tmp[full_itemR][1] + log_prob_new
                                                    full_new_psv_item = (bitVec_new, derivRule_new, new_extEdge)
                                                    if full_new_psv_item in oldItem2histLogP_tmp:
                                                        old_logProb = oldItem2histLogP_tmp[full_new_psv_item][1]
                                                        if old_logProb and cum_log_prob_new < old_logProb:
                                                            continue
                                                        # do nth if already handled by composition with semEmts above
                                                        if oldItem2histLogP_tmp[full_new_psv_item][2]:
                                                            if (full_itemC, full_itemL, full_itemR) == oldItem2histLogP_tmp[full_new_psv_item][0]\
                                                                and cum_log_prob_new < oldItem2histLogP_tmp[full_new_psv_item][1]:
                                                                continue
                                                    has_rule = True
    #                                                 print ("added", semEmt_type, surf, lOrR)
    #                                                 print (pred, new_dgtr_derivRules)
    #                                                 print ()
                                                    if forceSemEmt:
                                                        semEmt_added_BVs.add(new_bitVecs)
                    #                               # prop copula and semEmt
                                                    trgs_new =\
                                                        util.propagate_semEmtInfo_test(derivRule_new, new_semEmtTrgs)
                                                    oldItem2histLogP_tmp[full_new_psv_item] = (
                                                        (full_itemC, full_itemL, full_itemR),
                                                         cum_log_prob_new, surf, new_extEdges_dict, trgs_new, new_scope, new_indexFailCnt
                                                    )
                                                    bitVec2newPsvItems_tmp[bitVec_new][(derivRule_new, new_extEdge)] = new_extEdges_dict,\
                                                        trgs_new, new_scope, new_indexFailCnt
                                                    bitVec2rule2oldItems_tmp[bitVec_new][derivRule_new].add(new_extEdge)
            #                                         print (full_new_psv_item, 3)
                        if has_rule:
                            if bitVec_new in psvNodesBitVec2nbrsBitVec_tmp:
                                try:
                                    assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                        == (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]\
                                               | psvNodesBitVec2nbrsBitVec[bitVec_ccont]) & ~bitVec_new
                                except:
                                    print ("neighbour inconsistent 2")
                                    print (psvNodesBitVec2nbrsBitVec)
                                    print ((bitVec_new),(bitVec1), (bitVec2), (bitVec_ccont))
                                    print (bin(bitVec_new),bin(bitVec1), bin(bitVec2), bin(bitVec_ccont))
                                    # input()
                            else:
                                psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                    = (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]\
                                          | psvNodesBitVec2nbrsBitVec[bitVec_ccont]) & ~bitVec_new
    #                             print (bitVec_ccont, bitVec1, bitVec2, bitVec_new)
    #                             print (psvNodesBitVec2nbrsBitVec)
                                assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new] >= 0


        
        #binary
        no_items = 2
        for bitVec1 in bitVec2newPsvItems:
            if bitVec1 in bitVec2ccontCanon:
                continue
#             print (bin(bitVec1))
            isSingleNode1 =  util.is_power_of_two(bitVec1)
            for bitVec2 in bitVec2rule2oldItems:
#                 print (f"\t{bitVec2}")
#                 if bitVec1 == 4096 and bitVec2 == 8192:
#                     print ()
                if bitVec2 in bitVec2ccontCanon:
                    continue
                if bitVec1 == bitVec2 or bitVec1 & bitVec2 != 0:
                    continue
                if psvNodesBitVec2nbrsBitVec[bitVec1] & bitVec2 == 0:
                    assert psvNodesBitVec2nbrsBitVec[bitVec2] & bitVec1 == 0
                    continue
                assert psvNodesBitVec2nbrsBitVec[bitVec2] & bitVec1 != 0
                assert psvNodesBitVec2nbrsBitVec[bitVec1] & bitVec2 != 0
                if (bitVec1, bitVec2) in semEmt_added_BVs:
                    continue
                has_rule = False
                isSingleNode2 =  util.is_power_of_two(bitVec2)
                bitVec_new = bitVec1 | bitVec2
                for (psv_derivRule1, extEdges1), data1 in bitVec2newPsvItems[bitVec1].items():
                    psv_derivRule1_usp = lexCanon2usp.get(psv_derivRule1)
                    psv_derivRule1_prob = psv_derivRule1_usp
                    if not psv_derivRule1_usp: psv_derivRule1_prob = psv_derivRule1
                    for psv_derivRule2 in bitVec2rule2oldItems[bitVec2]:
                        psv_derivRule2_usp = lexCanon2usp.get(psv_derivRule2)
                        psv_derivRule2_prob = psv_derivRule2_usp
                        if not psv_derivRule2_usp: psv_derivRule2_prob = psv_derivRule2
                        lOrRs = [0, 0, 0]
                        if not (psv_derivRule2_usp or psv_derivRule1_usp):
                            if (psv_derivRule1, psv_derivRule2) in SHRG_rules2edges:
                                lOrRs[0] += 1
                            if (psv_derivRule2, psv_derivRule1) in SHRG_rules2edges:
                                lOrRs[0] += 2
                        if psv_derivRule1_usp:
                            if (psv_derivRule1_usp, psv_derivRule2) in SHRG_rules2edges:
                                lOrRs[1] += 1
                            if (psv_derivRule2, psv_derivRule1_usp) in SHRG_rules2edges:
                                lOrRs[1] += 2
                        if psv_derivRule2_usp:
                            if (psv_derivRule1, psv_derivRule2_usp) in SHRG_rules2edges:
                                lOrRs[2] += 1
                            if (psv_derivRule2_usp, psv_derivRule1) in SHRG_rules2edges:
                                lOrRs[2] += 2
                        
#                             pprint (SHRG_rules2edges)
#                         if bitVec_new == 196608:
#                             print (lOrRs)
#                             print (psv_derivRule1, psv_derivRule1_usp)
#                             print (psv_derivRule2, psv_derivRule2_usp)
#                             pprint (SHRG_rules2edges)
                        if not any(lOrRs):
                            continue
                            
                        if bitVec_new == 112 + 128:
                            pass
                            # print (lOrRs)
                            # print (psv_derivRule1, psv_derivRule1_usp)
                            # print (psv_derivRule2, psv_derivRule2_usp)
                        rules_tmp = [(psv_derivRule1, psv_derivRule2), (psv_derivRule1_usp, psv_derivRule2),
                                     (psv_derivRule1, psv_derivRule2_usp)]
                        for usp_idx, lOrR in enumerate(lOrRs):
                            if lOrR == 0: continue
                            psv_derivRule1_tmp, psv_derivRule2_tmp = rules_tmp[usp_idx]
                            directed = True
                            if lOrR == 3: directed = False
                            for extEdges2 in bitVec2rule2oldItems[bitVec2][psv_derivRule2]:
                                if bitVec2 in bitVec2newPsvItems and (psv_derivRule2, extEdges2) in bitVec2newPsvItems[bitVec2]\
                                    and bitVec1 > bitVec2:
                                    continue
                                extEdges_dicts1, semEmt_trgs1, scope1, indexFail1 = data1
                                extEdges_dicts2, semEmt_trgs2, scope2, indexFail2 = oldItem2histLogP_tmp[(bitVec2,
                                                                                                          psv_derivRule2, extEdges2)][3:]
                                two += 1
                                p = False
                                mode = 'B'
                                if lOrR == 2:
                                    shrgKey2psvItem = util.semCompMode2func[mode](dmrs_nxDG_ext, (psv_derivRule2_prob, psv_derivRule1_prob),
                                                                                (psv_derivRule2, psv_derivRule1),
                                                                        (psv_derivRule2_tmp, psv_derivRule1_tmp), None,
                                                                        (extEdges2, extEdges1),
                                                                            (extEdges_dicts2, extEdges_dicts1),
                                                                            (semEmt_trgs2, semEmt_trgs1), (scope2, scope1),
                                                                            (bitVec2, bitVec1),
                                                                            SHRG, edgeKey2lbl, dmrs_node2bitVec,
                                                                            no_items, mode = mode, p = p, directed = directed,
                                                                               return_exts = True,  node2pos = node2pos
, node_typed = node_typed)
                                else:
                                    shrgKey2psvItem = util.semCompMode2func[mode](dmrs_nxDG_ext, (psv_derivRule1_prob, psv_derivRule2_prob),
                                                                                (psv_derivRule1, psv_derivRule2),
                                                                        (psv_derivRule1_tmp, psv_derivRule2_tmp), None,
                                                                        (extEdges1, extEdges2),
                                                                            (extEdges_dicts1, extEdges_dicts2),
                                                                            (semEmt_trgs1, semEmt_trgs2), (scope1, scope2),
                                                                            (bitVec1, bitVec2),
                                                                            SHRG, edgeKey2lbl, dmrs_node2bitVec,
                                                                            no_items, mode = mode, p = p, directed = directed,
                                                                            return_exts = True,  node2pos = node2pos
, node_typed = node_typed)
                                if not shrgKey2psvItem:
                                    continue
#                                 if bitVec_new == 54 and "xp_brck-pr_c" in psv_derivRule1 or "xp_brck-pr_c" in psv_derivRule2:
#                                     print (derivRule_new)
        #                         if "mrk-nh_evnt_c" in psv_derivRule1 and "hd-cmp_u_c" in str(psv_derivRule2): print ("OK!")
                                two_suc += 1
                                for shrgKey, (new_bitVecs, new_semEmtTrgs, new_dgtr_scopes, new_extEdge,
                                              new_extEdges_dict, new_dgtr_extEdges, extEdgeKeyExts_list, interSubgrsEdges_keys) in shrgKey2psvItem.items():
                                    (new_dgtr_derivRules, interSubgrsEdges), new_dgtr_derivRules_exact, dgtr_derivRules_prob = shrgKey
                                    full_itemL = (new_bitVecs[0], new_dgtr_derivRules_exact[0], new_dgtr_extEdges[0])
                                    full_itemR = (new_bitVecs[1], new_dgtr_derivRules_exact[1], new_dgtr_extEdges[1])
                                    for derivRule_new, cnt in SHRG[(new_dgtr_derivRules, interSubgrsEdges)].items():
                                        new_scope = None
                                        new_indexFailCnt = 0
                                        if hookCheck:
                                            # check index availability here
                                            index_aval = util.index_aval_check(derivRule_new, extEdgeKeyExts_list,
                                                                              edgeKey2lbl, no_items)
                                            new_indexFailCnt = indexFail1 + indexFail2
                                            if not index_aval:
                                                contains_neg_edge = any([e in edge_set for e in neg_edges
                                                                         for edge_set in interSubgrsEdges_keys])
                                                if not contains_neg_edge:
                                                    new_indexFailCnt += 1
                                            if new_indexFailCnt > indexFailMax:
                                                if sum(new_bitVecs) == 8192 + 16384:
                                                    print (1234)
                                                continue
                                            ltop_aval, new_scope = util.ltop_aval_check(derivRule_new, extEdgeKeyExts_list, interSubgrsEdges,
                                                                                      edgeKey2lbl, new_dgtr_scopes, dmrsNode2scope, no_items, p = p)
                                            if not ltop_aval:
                                                if sum(new_bitVecs) ==  8192 + 16384:
                                                    print (12345)
                                                continue
                                            if final_ltop in new_dgtr_scopes and new_scope and new_scope != final_ltop:
                                                if sum(new_bitVecs) ==  8192 + 16384:
                                                    print (123)
                                                continue
                                        if usePHRG:
                                            try:
                                                log_prob_new = PHRG[(dgtr_derivRules_prob, interSubgrsEdges)][derivRule_new]
                                            except:
                                                print ((new_dgtr_derivRules, interSubgrsEdges))
                                                print (new_dgtr_derivRules_exact)
                                                print (dgtr_derivRules_prob)
                                                print (PHRG.get((dgtr_derivRules_prob, interSubgrsEdges)))
                                                print (SHRG.get((new_dgtr_derivRules, interSubgrsEdges)))
                                                print (SHRG_coarse.get(str((new_dgtr_derivRules, interSubgrsEdges))))
                                                print (derivRule_new)
#                                                 for i in PHRG:
#                                                     print (i)
#                                                     input()
                                                input()
                                        else:
                                            log_prob_new = dgtrs2binaryRule2logProb[(dgtr_derivRules_prob)][derivRule_new]
#                                         print (new_dgtr_derivRules_exact, new_dgtr_derivRules)
                                        cum_log_prob_new = oldItem2histLogP_tmp[full_itemL][1]\
                                            + oldItem2histLogP_tmp[full_itemR][1] + log_prob_new
                                        full_new_psv_item = (bitVec_new, derivRule_new, new_extEdge)
                                        if full_new_psv_item in oldItem2histLogP_tmp:
                                            old_logProb = oldItem2histLogP_tmp[full_new_psv_item][1]
                                            if old_logProb and cum_log_prob_new < old_logProb:
                                                continue
                                            # do nth if already handled by composition with semEmts above
                                            if oldItem2histLogP_tmp[full_new_psv_item][2]:
                                                if (full_itemL, full_itemR) == oldItem2histLogP_tmp[full_new_psv_item][0]:
                                                    continue
                                        has_rule = True
#                                         if bitVec_new == 54 and "xp_brck-pr_c" in psv_derivRule1 or "xp_brck-pr_c" in psv_derivRule2:
#                                             print (derivRule_new)
                                        trgs_new = util.propagate_semEmtInfo_test(derivRule_new, new_semEmtTrgs)
                                        oldItem2histLogP_tmp[full_new_psv_item] = (
                                            (full_itemL, full_itemR),
                                             cum_log_prob_new, None, new_extEdges_dict, trgs_new, new_scope, new_indexFailCnt
                                        )
                                        bitVec2newPsvItems_tmp[bitVec_new][(derivRule_new, new_extEdge)] = new_extEdges_dict,\
                                            trgs_new, new_scope, new_indexFailCnt
                                        bitVec2rule2oldItems_tmp[bitVec_new][derivRule_new].add(new_extEdge)
                if has_rule:
                    if bitVec_new in psvNodesBitVec2nbrsBitVec_tmp:
#                         print (psvNodesBitVec2nbrsBitVec_tmp)
                        assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                        == (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]) & ~bitVec_new
                    else:
                        psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                        = (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]) & ~bitVec_new
                    
                        assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new] >= 0

            
                        
                        
                        
#       binary (with one semEmt dgtr) with ccont pred
#       recognize the ccont pred without
        no_items = 2
        if not noSemEmt:
            for bitVec_ccont, (ccont_canon, ccont_extEdges, ccont_extEdges_dicts,
                              ccont_semEmtTrgs, ccont_scope, ccont_indexFail) in bitVec2ccontCanon.items():
                for bitVec1, newPsvItems in bitVec2newPsvItems.items():
                    if bitVec1 & bitVec_ccont != 0:
                        continue
                    if psvNodesBitVec2nbrsBitVec[bitVec_ccont] & bitVec1 == 0:
                        assert psvNodesBitVec2nbrsBitVec[bitVec1] & bitVec_ccont == 0
                        continue
                    has_rule = False
                    bitVec_new = bitVec1 | bitVec_ccont
    #                 if (bitVec_ccont, , bitVec1) not in semEmt_added_BVs:
                    for (psv_derivRule1, extEdges1), data1 in bitVec2newPsvItems[bitVec1].items():
                        psv_derivRule1_tmp = lexCanon2usp.get(psv_derivRule1) or psv_derivRule1
                        lOrR = 0
                        if (ccont_canon, psv_derivRule1_tmp, None) in SHRG_rules2edges:
                            lOrR += 1
                        if (ccont_canon, None, psv_derivRule1_tmp) in SHRG_rules2edges:
                            lOrR += 2
                        else: continue
                        extEdges_dicts1, semEmt_trgs1, scope1, indexFail1 = data1
                        directed = True
                        p = False
                        if lOrR == 3: directed = False
    #                     print (ccont_canon, psv_derivRule1)#, derivRule_new)
                        semEmt_trgs2 = (None, None, None)
                        mode = 'BPsemEmt'
                        if lOrR == 2:
                            shrgKey2psvItem = util.semCompMode2func[mode](dmrs_nxDG_ext, _,
                                                                            (ccont_canon, None, psv_derivRule1),
                                                                        (ccont_canon, None, psv_derivRule1_tmp), None,
                                                                        (ccont_extEdges, (), extEdges1),
                                                                        (ccont_extEdges_dicts, ({}, {}), extEdges_dicts1),
                                                                        (ccont_semEmtTrgs, semEmt_trgs2, semEmt_trgs1),
                                                                        (ccont_scope, None, scope1),
                                                                        (bitVec_ccont, None, bitVec1),
                                                                        SHRG, edgeKey2lbl, dmrs_node2bitVec,
                                                                        no_items, mode = mode, p = p, directed = directed,
                                                                        return_exts = True, node2pos = node2pos, node_typed = node_typed)
                        else:
                            shrgKey2psvItem = util.semCompMode2func[mode](dmrs_nxDG_ext, _,
                                                                            (ccont_canon, psv_derivRule1, None),
                                                                        (ccont_canon, psv_derivRule1_tmp, None), None,
                                                                        (ccont_extEdges, extEdges1, ()),
                                                                        (ccont_extEdges_dicts, extEdges_dicts1, ({}, {})),
                                                                        (ccont_semEmtTrgs, semEmt_trgs1, semEmt_trgs2),
                                                                        (ccont_scope, scope1, None),
                                                                        (bitVec_ccont, bitVec1, None),
                                                                        SHRG, edgeKey2lbl, dmrs_node2bitVec,
                                                                        no_items, mode = mode, p = p, directed = directed,
                                                                        return_exts = True, node2pos = node2pos, node_typed = node_typed)
                        if not shrgKey2psvItem:
                            continue
                        twofive_suc += 1
                        for shrgKey, (new_bitVecs, new_semEmtTrgs, new_dgtr_scopes, new_extEdge,
                                      new_extEdges_dict, new_dgtr_extEdges, extEdgeKeyExts_list, interSubgrsEdges_keys) in shrgKey2psvItem.items():
                            (new_dgtr_derivRules, interSubgrsEdges), new_dgtr_derivRules_exact = shrgKey
                            for derivRule_new, cnt in SHRG[(new_dgtr_derivRules, interSubgrsEdges)].items():
                                # do not check index availability here
                                cum_log_prob_new = oldItem2histLogP_tmp[(bitVec1, psv_derivRule1, extEdges1)][1]
                                full_new_psv_item = (bitVec_new, psv_derivRule1, new_extEdge)
                                if full_new_psv_item in oldItem2histLogP_tmp:
                                    old_logProb = oldItem2histLogP_tmp[full_new_psv_item][1]
                                    if old_logProb and cum_log_prob_new < old_logProb:
                                        continue
                                    # do nth if already handled by composition with semEmts above
                                    if oldItem2histLogP_tmp[full_new_psv_item][2]:
                                        if ((bitVec1, psv_derivRule1, extEdges1),) == oldItem2histLogP_tmp[full_new_psv_item][0]:
                                            continue
                                has_rule = True
                                # prop copula and semEmt
                                trgs_new = util.propagate_semEmtInfo_test(derivRule_new, new_semEmtTrgs)
                                oldItem2histLogP_tmp[(bitVec_new, psv_derivRule1, new_extEdge)]\
                                    = (((bitVec1, psv_derivRule1, extEdges1),), cum_log_prob_new,  None,
                                       new_extEdges_dict, trgs_new, scope1, indexFail1)
                                bitVec2newPsvItems_tmp[bitVec_new][(psv_derivRule1, new_extEdge)] = new_extEdges_dict,\
                                        trgs_new, scope1, indexFail1
                                bitVec2rule2oldItems_tmp[bitVec_new][psv_derivRule1].add(new_extEdge)

                    if has_rule:
                        if bitVec_new in psvNodesBitVec2nbrsBitVec_tmp:
                            assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                == (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec_ccont]) & ~bitVec_new
                        else:
                            psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                = (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec_ccont]) & ~bitVec_new
                            assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new] >= 0
                        
        # binary (with no semEmt dgtr) with ccont pred
        no_items = 3
        for bitVec_ccont, (ccont_canon, ccont_extEdges, ccont_extEdges_dicts,
                           ccont_semEmtTrgs, ccont_scope, ccont_indexFail) in bitVec2ccontCanon.items():
            for bitVec1 in bitVec2newPsvItems:
                if bitVec1 in bitVec2ccontCanon:
                    continue
                for bitVec2 in bitVec2rule2oldItems:
                    if bitVec2 in bitVec2ccontCanon:
                        continue
                    if bitVec1 & bitVec2 != 0 or bitVec1 & bitVec_ccont or bitVec2 & bitVec_ccont != 0:
                        continue
                    if sum([1 if a & b != 0 else 0
                            for a,b in [[psvNodesBitVec2nbrsBitVec[bitVec_ccont], bitVec1],
                                        [psvNodesBitVec2nbrsBitVec[bitVec1], bitVec2],
                                        [psvNodesBitVec2nbrsBitVec[bitVec2], bitVec_ccont]]]) < 2:
                        continue
                    has_rule = False
                    bitVec_new = bitVec2 | bitVec_ccont | bitVec1
                    if (bitVec_ccont, bitVec1, bitVec2) in semEmt_added_BVs: continue
                    for (psv_derivRule1, extEdges1), data1 in bitVec2newPsvItems[bitVec1].items():
                        psv_derivRule1_tmp = lexCanon2usp.get(psv_derivRule1) or psv_derivRule1
                        extEdges_dicts1, semEmt_trgs1, scope1, indexFail1 = data1
                        for psv_derivRule2 in bitVec2rule2oldItems[bitVec2]:
                            psv_derivRule2_tmp = lexCanon2usp.get(psv_derivRule2) or psv_derivRule2
#                             print (psv_derivRule1_tmp, psv_derivRule2_tmp)
                            lOrR = 0
                            if (ccont_canon, psv_derivRule1_tmp, psv_derivRule2_tmp) in SHRG_rules2edges:
                                lOrR += 1
                            if (ccont_canon, psv_derivRule2_tmp, psv_derivRule1_tmp) in SHRG_rules2edges:
                                lOrR += 2
                            if lOrR == 0:
                                continue
#                             print ((ccont_canon, psv_derivRule2_tmp, psv_derivRule1_tmp) )
                            for extEdges2 in bitVec2rule2oldItems[bitVec2][psv_derivRule2]:
                                if bitVec2 in bitVec2newPsvItems and (psv_derivRule2, extEdges2) in bitVec2newPsvItems[bitVec2]\
                                    and bitVec1 > bitVec2:
                                    continue
                                extEdges_dicts2, semEmt_trgs2, scope2, indexFail2 = oldItem2histLogP_tmp[(bitVec2, psv_derivRule2, extEdges2)][3:]
                                new_indexFailCnt = indexFail1 + indexFail2
                                p = False
                                three += 1
                                directed = True
                                if lOrR == 3: directed = False
                                mode = 'BP'
                                if lOrR == 2:
                                    shrgKey2psvItem = util.semCompMode2func[mode](dmrs_nxDG_ext, _,
                                                                        (ccont_canon, psv_derivRule2, psv_derivRule1),
                                                                        (ccont_canon, psv_derivRule2_tmp, psv_derivRule1_tmp), None,
                                                                        (ccont_extEdges, extEdges2, extEdges1),
                                                                        (ccont_extEdges_dicts, extEdges_dicts2, extEdges_dicts1),
                                                                            (ccont_semEmtTrgs, semEmt_trgs2, semEmt_trgs2),
                                                                            (ccont_scope, scope2, scope1),
                                                                            (bitVec_ccont, bitVec2, bitVec1),
                                                                            SHRG, edgeKey2lbl, dmrs_node2bitVec,
                                                                            no_items, mode = mode, p = p, directed = directed,
                                                                               return_exts = True, node2pos = node2pos, node_typed = node_typed)
                                else:
                                    shrgKey2psvItem = util.semCompMode2func[mode](dmrs_nxDG_ext, _,
                                                                        (ccont_canon, psv_derivRule1, psv_derivRule2),
                                                                        (ccont_canon, psv_derivRule1_tmp, psv_derivRule2_tmp), None,
                                                                        (ccont_extEdges, extEdges1, extEdges2),
                                                                        (ccont_extEdges_dicts, extEdges_dicts1, extEdges_dicts2),
                                                                            (ccont_semEmtTrgs, semEmt_trgs1, semEmt_trgs2),
                                                                            (ccont_scope, scope1, scope2),
                                                                            (bitVec_ccont, bitVec1, bitVec2),
                                                                            SHRG, edgeKey2lbl, dmrs_node2bitVec,
                                                                            no_items, mode = mode, p = p, directed = directed,
                                                                               return_exts = True, node2pos = node2pos, node_typed = node_typed)
                                if not shrgKey2psvItem:
                                    continue
                                three_suc += 1
                                for shrgKey, (new_bitVecs, new_semEmtTrgs, new_dgtr_scopes, new_extEdge,
                                              new_extEdges_dict, new_dgtr_extEdges, extEdgeKeyExts_list,
                                             interSubgrsEdges_keys) in shrgKey2psvItem.items():
                                    (new_dgtr_derivRules, interSubgrsEdges), new_dgtr_derivRules_exact = shrgKey
                                    full_itemC = (new_bitVecs[0], new_dgtr_derivRules_exact[0], new_dgtr_extEdges[0])
                                    full_itemL = (new_bitVecs[1], new_dgtr_derivRules_exact[1], new_dgtr_extEdges[1])
                                    full_itemR = (new_bitVecs[2], new_dgtr_derivRules_exact[2], new_dgtr_extEdges[2])
                                    for derivRule_new, cnt in SHRG[(new_dgtr_derivRules, interSubgrsEdges)].items():
                                        new_scope = None
                                        new_indexFailCnt = 0
                                        if hookCheck:
                                            index_aval = util.index_aval_check(derivRule_new, extEdgeKeyExts_list,
                                                                              edgeKey2lbl, no_items)
                                            if not index_aval:
                                                contains_neg_edge = any([e in edge_set for e in neg_edges
                                                                         for edge_set in interSubgrsEdges_keys])
                                                if not contains_neg_edge:
                                                    new_indexFailCnt += 1
                                            if new_indexFailCnt > indexFailMax:
                                                continue
                                            ltop_aval, new_scope = util.ltop_aval_check(derivRule_new, extEdgeKeyExts_list, interSubgrsEdges,
                                                                                  edgeKey2lbl, new_dgtr_scopes, dmrsNode2scope, 
                                                                                        no_items, p)
                                            if not ltop_aval:
                                                continue
                                            if final_ltop in new_dgtr_scopes and new_scope and new_scope != final_ltop:
                                                continue
                                        if usePHRG:
                                            log_prob_new = PHRG[(new_dgtr_derivRules, interSubgrsEdges)][derivRule_new]
                                        else:
                                            log_prob_new = dgtrs2binaryRule2logProb[new_dgtr_derivRules[1:]][derivRule_new]
                                        cum_log_prob_new = oldItem2histLogP_tmp[full_itemL][1]\
                                            + oldItem2histLogP_tmp[full_itemR][1] + log_prob_new
                                        full_new_psv_item = (bitVec_new, derivRule_new, new_extEdge)
                                        if full_new_psv_item in oldItem2histLogP_tmp:
                                            old_logProb = oldItem2histLogP_tmp[full_new_psv_item][1]
                                            if old_logProb and cum_log_prob_new < old_logProb:
                                                continue
                                            # do nth if already handled by composition with semEmts above
                                            if oldItem2histLogP_tmp[full_new_psv_item][2]:
                                                if (full_itemC, full_itemL, full_itemR) == oldItem2histLogP_tmp[full_new_psv_item][0]:
                                                    continue
#                                         print (derivRule_new)
                                        has_rule = True
        #                               # prop copula and semEmt
                                        trgs_new =\
                                            util.propagate_semEmtInfo_test(derivRule_new, new_semEmtTrgs)
                                        oldItem2histLogP_tmp[full_new_psv_item] = (
                                            (full_itemC, full_itemL, full_itemR),
                                             cum_log_prob_new, None, new_extEdges_dict, trgs_new, new_scope, new_indexFailCnt
                                        )
                                        bitVec2newPsvItems_tmp[bitVec_new][(derivRule_new, new_extEdge)] = new_extEdges_dict,\
                                            trgs_new, new_scope, new_indexFailCnt
                                        bitVec2rule2oldItems_tmp[bitVec_new][derivRule_new].add(new_extEdge)
#                                         print (full_new_psv_item, 3)
                    if has_rule:
                        if bitVec_new in psvNodesBitVec2nbrsBitVec_tmp:
                            try:
                                assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                    == (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]\
                                           | psvNodesBitVec2nbrsBitVec[bitVec_ccont]) & ~bitVec_new
                            except:
                                print ("neighbour inconsistent 3")
                                print (psvNodesBitVec2nbrsBitVec)
                                print ((bitVec_new),(bitVec1), (bitVec2), (bitVec_ccont))
                                print (bin(bitVec_new),bin(bitVec1), bin(bitVec2), bin(bitVec_ccont))
                                # input()
                        else:
                            psvNodesBitVec2nbrsBitVec_tmp[bitVec_new]\
                                = (psvNodesBitVec2nbrsBitVec[bitVec1] | psvNodesBitVec2nbrsBitVec[bitVec2]\
                                      | psvNodesBitVec2nbrsBitVec[bitVec_ccont]) & ~bitVec_new
#                             print (bitVec_ccont, bitVec1, bitVec2, bitVec_new)
#                             print (psvNodesBitVec2nbrsBitVec)
                            assert psvNodesBitVec2nbrsBitVec_tmp[bitVec_new] >= 0
        
        
        bitVec2newPsvItems = bitVec2newPsvItems_tmp
        oldItem2histLogP = oldItem2histLogP_tmp
        bitVec2rule2oldItems = bitVec2rule2oldItems_tmp
        psvNodesBitVec2nbrsBitVec = psvNodesBitVec2nbrsBitVec_tmp

    
    ruleCheckCnts = one, one_suc, two, two_suc, twofive, twofive_suc, three, three_suc, cop, cop_suc, sem, sem_suc
    return Scompleted, completed, oldItem2histLogP, bitVec2rule2oldItems, ruleCheckCnts

def score_with_bleu(pred, truth):
    try:
        return sacrebleu.sentence_bleu(pred, truth) #python
    except:
        return sacrebleu.sentence_bleu(pred, [truth]) # pypy

def realize_dmrs2(snt_id, dmrs_nxDG, dmrs_nxDG_ancAdj, true_deriv_nxDG = None, true_sentence = None, debug = True, drawTree = False):
    
    def post_process(realized_list):
        realized = ""
        if realized_list == []:
            return ""
        realized_list[0] = realized_list[0].capitalize()
        for idx, surface in enumerate(realized_list):
            if idx == 0:
                realized += surface.capitalize()
                continue
            if surface in ['un-','re-','mis-']:
                realized += " {}".format(surface)
                continue
            if realized_list[idx - 1][-1] == '-':
                realized += surface
                continue
            if surface == "'s" or surface.startswith("'") and surface in util.copula:
                realized += surface
                continue
            if surface.lower() == 'a' and len(realized_list) -1 >= idx + 1\
                and realized_list[idx + 1][0].lower() in ['a', 'e', 'i', 'o', 'u']:
                realized += " {}n".format(surface)
                continue
            else:
                realized += " {}".format(surface)
                continue
        realized += '.'
        return realized
            
    def get_realized(toBeRealized):
        # preTermCanon2bTagSurface2cnt
        realized_list = []
        unrealized_list = []
#         pprint (toBeRealized)
        for (CanonOrPreterm, add_surf_prop) in toBeRealized:
            hvPreterm_surfaceCnt = None
            if CanonOrPreterm == "<,>":
                surface = ','
            else:
                if isinstance(CanonOrPreterm, str):
                    hvPreterm_surfaceCnt = preTermAnno2surface2cnt.get(CanonOrPreterm) or\
                        preTerm2surface2cnt.get(CanonOrPreterm)
                if not hvPreterm_surfaceCnt:
                    if isinstance(CanonOrPreterm, str):
                        print (CanonOrPreterm, ": preterm surface not recorded")
                        surface = CanonOrPreterm.split("/")[0].split("_")[0]
                    elif isinstance(CanonOrPreterm, tuple):
                        surface = util.get_surfaceFromCanon(preTermCanon2surface2cnt, preTermCanon2bTagSurface2cnt,
                                                            CanonOrPreterm, add_surf_prop)
                else:
                    surface, cnt = hvPreterm_surfaceCnt.most_common()[0]
#             print (type(surfaces))
#             print (surfaces.most_common()[0][0])
#             print (surfaces)
            if surface == None:
                unrealized_list.append(CanonOrPreterm)
            elif surface != "":
                realized_list.append(surface)
        realized = post_process(realized_list)
        return realized, unrealized_list
    
    def get_deriv(curr_key, curr_bitVec, curr_node, par_node = None, edge_lbl = "", build_tree = True, record_bitVecRule = False):
#         print (curr_key, curr_prob, curr_bitVec, curr_node)
        nonlocal bitVec2predRule
        bitVec, deriv_rule, ext_edges = curr_key
        if deriv_rule[0] == "UC":
            unaryChain = list(deriv_rule[1:]) #top-to-bottom
            unaryChain.reverse()
            if build_tree:
                p = par_node
                for r in unaryChain:
                    restored_deriv.add_node(curr_node)
                    if p:
                        restored_deriv.add_edge(p, curr_node, label = edge_lbl)
                        p = curr_node
                    restored_deriv.nodes[curr_node]['entity'] = r
                    curr_node = curr_node * 2
                curr_node = curr_node/2
            if record_bitVecRule:
                for r in unaryChain:
                    if isinstance(r, str):
                        bitVec2predRule[bitVec].append(r.split("&")[0].split("^")[0]) 
            hist, curr_prob, surfaces, *_ = oldItem2histLogP[(bitVec, unaryChain[-1], ext_edges)]
            
        else:
            if build_tree:
                restored_deriv.add_node(curr_node)
                if par_node:
                    restored_deriv.add_edge(par_node, curr_node, label = edge_lbl)
                restored_deriv.nodes[curr_node]['entity'] = deriv_rule
            if curr_key[2] == "surface_canon":
                hist = None
                curr_prob = 0
                return [deriv_rule]
            if record_bitVecRule and isinstance(deriv_rule, str):
                bitVec2predRule[bitVec].append(deriv_rule.split("&")[0].split("^")[0])
                
            hist, curr_prob, surfaces, *_ = oldItem2histLogP[curr_key]
      
        if build_tree:
            if surfaces:
                restored_deriv.nodes[curr_node]['surfaces'] = surfaces

            restored_deriv.nodes[curr_node]['sumLogProb'] = curr_prob
        if not surfaces:
            surfaces = ["<left>", "<right>"]
            if deriv_rule in rule2commaProb:
                if rule2commaProb[deriv_rule] > commaProb_min:
                    surfaces = ["<left>", "<,>", "<right>"]
        # print (curr_node, deriv_rule)
        if len(hist) == 3:
            l = get_deriv(hist[1], hist[1][0], curr_node * 2, curr_node, 'L', build_tree = build_tree,
                          record_bitVecRule = record_bitVecRule)
            r = get_deriv(hist[2], hist[2][0], curr_node * 2 + 1, curr_node, 'R', build_tree = build_tree,
                          record_bitVecRule = record_bitVecRule)
            return list(chain.from_iterable(
                            [l
                                if surface == "<left>"
                            else r
                                if surface == "<right>"
                            else [("<,>", None)]
                                if surface == "<,>"
                            else
                                [(surface, None)]
                            for surface in surfaces]))
        if len(hist) == 2:
            return list(chain.from_iterable(
                    [get_deriv(hist[0], hist[0][0], curr_node * 2, curr_node,  'L', build_tree = build_tree,
                          record_bitVecRule = record_bitVecRule)
                    if surface == "<left>"
                    else get_deriv(hist[1], hist[1][0], curr_node * 2 + 1, curr_node, 'R', build_tree = build_tree,
                          record_bitVecRule = record_bitVecRule)
                    if surface == "<right>"
                    else [("<,>", None)]
                    if surface == "<,>"
                    else [(surface, None)]
                     for surface in surfaces]))

        elif len(hist) == 1:
            return get_deriv(hist[0],
                              hist[0][0], curr_node * 2, curr_node, 'U', build_tree = build_tree,
                              record_bitVecRule = record_bitVecRule)
    
    def init_chart(dmrs_nxDG):
        nonlocal dmrs_nxDG_ext, edgeKey2lbl, dmrs_node2bitVec, dmrsNode2scope, final_ltop, pred2edgesBV2key2lbl, neg_edges, node2pos
        dmrs_nxDG_ext = dmrs_nxDG.copy()
        # augment edge labels for external node recognition
        for edge in dmrs_nxDG.edges(keys = True, data = 'label'):
            edgeKey2lbl[edge[:3]] = edge[-1]
        for edge in dmrs_nxDG_ext.edges(keys = True):
            src_pos = ""
            targ_pos = ""
            if node_typed:
                _, src_pos = util.get_lemma_pos(dmrs_nxDG_ext.nodes[edge[0]])
                _, targ_pos = util.get_lemma_pos(dmrs_nxDG_ext.nodes[edge[1]])
                src_pos = util.unknown2pos.get(src_pos) or src_pos
                targ_pos = util.unknown2pos.get(targ_pos) or targ_pos
                node2pos[edge[0]] = src_pos
                node2pos[edge[1]] = targ_pos
            dmrs_nxDG_ext.edges[edge]['label'] = '#0{}-src-'.format(src_pos) + dmrs_nxDG_ext.edges[edge]['label'] + '-targ-{}#0'.format(targ_pos)
            # print (dmrs_nxDG_ext.edges[edge]['label'])
        # assign ordered ext nodes to itself
        order_char_val = list(range(35,39)) + list(range(40,47)) + list(range(48,58)) + list(range(65,91)) + list(range(97,122))
        order_char_val_idx = 0
        for node in dmrs_nxDG_ext.nodes():
            dmrs_nxDG_ext.nodes[node]['order_char'] = chr(order_char_val[order_char_val_idx])
            dmrs_nxDG_ext.nodes[node]['order_int'] = order_char_val_idx
            dmrs_node2bitVec[node] = 2**order_char_val_idx
            order_char_val_idx += 1
        for node1 in dmrs_nxDG_ext.nodes():
            for node2 in dmrs_nxDG_ext.nodes():
                if node1 == node2: continue
                if dmrs_nxDG_ext.has_edge(node1,node2) or dmrs_nxDG_ext.has_edge(node2,node1):
                    psvNodesBitVec2nbrsBitVec[2**dmrs_nxDG_ext.nodes[node1]['order_int']]\
                        += 2**dmrs_nxDG_ext.nodes[node2]['order_int']
#                     dmrs_node2nbrs[node1].add(node2)
#                     dmrs_node2nbrs[node2].add(node1)
        scope_idx = 0
        for src, targ, lbl in dmrs_nxDG_ext.edges(data = 'label'):
            if "/EQ" in lbl:
                if src in dmrsNode2scope or targ in dmrsNode2scope:
                    if src in dmrsNode2scope: dmrsNode2scope[targ] = dmrsNode2scope[src]
                    elif targ in dmrsNode2scope: dmrsNode2scope[src] = dmrsNode2scope[targ]
                else:
                    dmrsNode2scope[src] = scope_idx
                    dmrsNode2scope[targ] = scope_idx
                    scope_idx += 1
        for node in dmrs_nxDG_ext.nodes():
            if node not in dmrsNode2scope:
                dmrsNode2scope[node] = scope_idx
                scope_idx += 1
        final_ltop = dmrsNode2scope[dmrs_nxDG.graph['top']]
        # build predSemEmt trg index
        for src, targ, key, lbl in dmrs_nxDG.edges(keys = True, data = 'label'):
            # record neg -1//H-> X
            if dmrs_nxDG.nodes[src]['instance'] == 'neg' and 'ARG1/H' in lbl:
                neg_edges.add((src, targ, key))
            edge2surf2logProb = predTrg2edge2surf2logProb.get(dmrs_nxDG.nodes[src]['instance'])
            if edge2surf2logProb:
#                 print (lbl)
#                 print (edge2surf2logProb)
                if lbl in edge2surf2logProb:
                    
                    for (derivRule, surf), logProb in edge2surf2logProb[lbl].items():
                        pred2edgesBV2key2lbl[dmrs_nxDG.nodes[src]['instance']][(2**dmrs_nxDG_ext.nodes[src]['order_int'],
                                                                      2**dmrs_nxDG_ext.nodes[targ]['order_int'])][(src, targ, key)] = lbl
#                     edge2rule2surfLogProb[(src, targ, key)][derivRule] = (surf, logProb)
#         pprint (pred2edgesBV2key2lbl)
        
            
                                                           
    def replace_preterminals(snt_id, dmrs_nxDG_orig, dmrs_nxDG_ext, dmrs_nxDG_ancAdj,
                             true_deriv_nxDG, true_sentence, debug = True):
        global rule2commaProb, startRule2cnt, ergRule2cnt, dmrsSubgrCanon2bTag2cnt, preTermCanon2bTagSurface2cnt, preTerm2surface2cnt, preTermAnno2surface2cnt, preTermCanon2surface2cnt, preTermCanonUsps2cnt, dgtrs2unaryRule2cnt, dgtrs2binaryRule2cnt, PHRG, SHRG, SHRG_coarse, ccontCanon2intSubgrEdges2cnt, canon2cnt, canon_usp2cnt, dgtrs2unaryRule2logProb, dgtrs2binaryRule2logProb, eqAnc_semiCanons, eqAnc_semiCanons_usp, ccont_semiCanons
        global exactCanonsToMatch, ccontCanonsToMatch, uspCanonsToMatch
        nonlocal node2pos
        # Match Subgraphs with Canonical form
        all_sizei_nodesSubgraph, all_sizei_nodesSubgraph_usp, _, canon2canon_usp, _\
            = util.gen_all_maxsizek_subgraph(dmrs_nxDG_ext, max_subgraph_size,
                                             sentence = None, subgraphs_semicanon = eqAnc_semiCanons,
                                             subgraphs_semicanon_usp = eqAnc_semiCanons_usp ,
                                             extract_surfacemap = False, lexicalized = lexicalized)
        all_sizei_nodesSubgraphs = [all_sizei_nodesSubgraph, all_sizei_nodesSubgraph, all_sizei_nodesSubgraph_usp]
        exactUspCanonsToMatch = [exactCanonsToMatch, ccontCanonsToMatch, uspCanonsToMatch]
        # exact, usp subgrapoh and single node preterminal for 0, 1, 2 resp.
        for itr in [0,1,2,3]:
            if itr in [0, 1, 2]:
                for i in range(max_subgraph_size,1,-1):
                    no_items = i
                    if all_sizei_nodesSubgraphs[itr][i]:
#                         print (all_sizei_nodesSubgraphs[0])
#                         input() 
                        for sorted_nodes, subgraph_canon, _ in all_sizei_nodesSubgraphs[itr][i]:
                            surface_canon = None
                            subgraph_canon, surface_canon = subgraph_canon
#                             print (itr, subgraph_canon, surface_canon)
                            if (itr in [0] and surface_canon in exactUspCanonsToMatch[itr]\
                                    or itr in [1,2] and subgraph_canon in exactUspCanonsToMatch[itr]):
                                if itr in [0,1] or subgraph_canon in filtered_preTermCanonUsp:
                                    bv_val = sum([2**dmrs_nxDG_ext.nodes[node]['order_int'] for node in sorted_nodes])
# #                                     bs = str(BitVector(intVal = bv_val))
#                                     if sum([2**dmrs_nxDG_ext.nodes[node]['order_int'] for node in sorted_nodes]) == 8192 + 16384:
#                                         print (subgraph_canon)
                                    bs = bv_val
                                    mode = 'SubgrTag'
                                    new_extEdge, new_extEdges_dicts = util.semCompMode2func[mode](dmrs_nxDG_ext, _,
                                                                            subgraph_canon, subgraph_canon, sorted_nodes,
                                                                            None, None, None, None, None, SHRG,
                                                                                edgeKey2lbl, dmrs_node2bitVec,
                                                                           no_items, mode = mode, node2pos = node2pos, node_typed = node_typed)
                                    if itr in [0,2]:
                                        if itr == 0:
                                            exact_canon = surface_canon
                                            if dmrsSubgrCanon2bTag2cnt.get(exact_canon):
                                                # assume one derivation of terminal
                                                subgraph_canon = dmrsSubgrCanon2bTag2cnt.get(exact_canon).most_common()[0][0]
#                                                 print (subgraph_canon, "bbbb!")
                                            else:
                                                subgraph_canon = canon2canon_usp[subgraph_canon]
#                                             print ("0!!!!!!!", exact_canon)
                                        # get semEmts info
                                        cop_trg_final, prtcl_trg_final, compl_trg_final = None, None, None
                                        for node in sorted_nodes:
                                            # get semEmts info
                                            # prtcl trg
                                            prtcl = util.get_pred_prtcl(dmrs_nxDG_orig.nodes[node]['instance'])
                                            if prtcl:
                                                prtcl_trg_final = util.get_prtcl_trg(dmrs_nxDG_orig, node, prtcl)
#                                                 print ("prtcl trg: ", prtcl_trg_final)
                                            # copula trg
                                            copula_trg, tense = util.get_copula_trg(dmrs_nxDG_orig, node)
                                            if not cop_trg_final or tense != 'UNTENSED' and copula_trg:
                                                cop_trg_final = copula_trg
                                            # compl trg
                                            compl_trg = util.get_compl_trg(dmrs_nxDG_orig, node)
                                            if compl_trg:
                                                compl_trg_final = compl_trg
                                        trgs = cop_trg_final, prtcl_trg_final, compl_trg_final
#                                                 print ("copula trg: ", copula_trg)
                                                
#                                             semEmt = util.get_pred_prtcl(dmrs_nxDG_orig.nodes[node]['instance'])
# #                                             print (semEmt)
#                                             # copula trg
#                                             copula_trg, tense = util.get_copula_trg(dmrs_nxDG_orig, node)
#                                             if not cop_trg_final or tense != 'UNTENSED' and copula_trg:
#                                                 cop_trg_final = copula_trg
#                                             # general semEmt trg (elif (in training) to if during test to improve coverage)
#                                             if semEmt:
#                                                 if dmrs_nxDG_orig.nodes[node]['instance'] == 'parg_d':
#                                                     semEmt_trg_final = subgraph_canon
#                                                 else:
#                                                     semEmt_trg_final = exact_canon
                                        if itr == 2: lexCanon2usp[subgraph_canon] = subgraph_canon
                                        bitVec2rule2oldItems[bs][subgraph_canon].add(new_extEdge)
                                        oldItem2histLogP[(bs, subgraph_canon, new_extEdge)] =\
                                            (((bs, (surface_canon, None), "surface_canon"),), 0, None, new_extEdges_dicts,
                                            trgs, None, 0)
                                        bitVec2newPsvItems[bs][(subgraph_canon, new_extEdge)]\
                                            = new_extEdges_dicts, trgs, None, 0
                                    elif itr == 1:
                                        cop_trg_final, prtcl_trg_final, compl_trg_final = None, None, None
                                        trgs = cop_trg_final, prtcl_trg_final, compl_trg_final
                                        if bs in bitVec2ccontCanon:
                                            assert bitVec2ccontCanon[bs] == (subgraph_canon, new_extEdge,
                                                                            new_extEdges_dicts, trgs, None, 0)
                                        else:
                                            bitVec2ccontCanon[bs] = (subgraph_canon, new_extEdge,
                                                                    new_extEdges_dicts, trgs, None, 0)
                                    for node in sorted_nodes:
                                        psvNodesBitVec2nbrsBitVec[bs]\
                                            |= psvNodesBitVec2nbrsBitVec[2**dmrs_nxDG_ext.nodes[node]['order_int']]
                                    for node in sorted_nodes:
                                        psvNodesBitVec2nbrsBitVec[bs] -= 2**dmrs_nxDG_ext.nodes[node]['order_int']
            elif itr in [3]:
                no_items = 1
#                 print (bitVec2ccontCanon)
                for node in dmrs_nxDG_ext.nodes():
                    preTermCanon = None # in vocab
                    surface_canonical_form, sorted_nodes, _\
                        = util.get_canonical_form(dmrs_nxDG_ext.subgraph([node]),
                                            dmrs_nxDG_ext, extract_surfacemap = False, forSurfGen = True)
                    canonical_form, sorted_nodes, _\
                        = util.get_canonical_form(dmrs_nxDG_ext.subgraph([node]),
                                            dmrs_nxDG_ext, extract_surfacemap = False,
                                                 lexicalized = lexicalized)
                    usp_canonical_form, sorted_nodes, _\
                        = util.get_canonical_form(dmrs_nxDG_ext.subgraph([node]),
                                            dmrs_nxDG_ext, extract_surfacemap = False,
                                            underspecLemma = True, underspecCarg = True,
                                                  lexicalized = lexicalized)
                    if str(surface_canonical_form) in ccontCanon2intSubgrEdges2cnt:
                        preTermCanon = surface_canonical_form
                    elif str(canonical_form) in preTermCanonUsps2cnt:
                        preTermCanon = canonical_form
                        lexCanon2usp[canonical_form] = usp_canonical_form
                        # print (preTermCanon, 2)
                    else:
#                         print (preTermCanon, usp_canonical_form, 7)
                        preTermCanon = usp_canonical_form
                        # print (preTermCanon, usp_canonical_form, 3)
#                     print (preTermCanon)
                    bv_val = 2**dmrs_nxDG_ext.nodes[node]['order_int']
                    bs = bv_val
                    mode = 'Tag'
                    new_extEdge, new_extEdges_dicts = util.semCompMode2func[mode](dmrs_nxDG_ext, _, preTermCanon, preTermCanon,
                                                                                [node], None, None, None,
                                                                             None, None, SHRG, edgeKey2lbl, dmrs_node2bitVec,
                                                                                no_items, mode = mode, node2pos = node2pos, node_typed = node_typed)
                    # get semEmts info
                    cop_trg_final, prtcl_trg_final, compl_trg_final = None, None, None
                    # prtcl trg
                    prtcl = util.get_pred_prtcl(dmrs_nxDG_orig.nodes[node]['instance'])
                    if prtcl:
                        prtcl_trg_final = util.get_prtcl_trg(dmrs_nxDG_orig, node, prtcl)
#                         print ("prtcl trg: ", prtcl_trg_final)
                    # copula trg
                    copula_trg, tense = util.get_copula_trg(dmrs_nxDG_orig, node)
                    if not cop_trg_final or tense != 'UNTENSED' and copula_trg:
                        cop_trg_final = copula_trg
                    # compl trg
                    compl_trg = util.get_compl_trg(dmrs_nxDG_orig, node)
                    if compl_trg:
                        compl_trg_final = compl_trg
                    trgs = cop_trg_final, prtcl_trg_final, compl_trg_final
#                         print ("copula trg: ", copula_trg)
                        
#                     # copula trg
#                     copula_trg, tense = util.get_copula_trg(dmrs_nxDG_orig, node)
#                     if not cop_trg_final or tense != 'UNTENSED' and copula_trg:
#                         cop_trg_final = copula_trg
#                     # general semEmt trg  (elif (in training) to if during test to improve coverage)
#                     if semEmt:
#                         if dmrs_nxDG_orig.nodes[node]['instance'] == 'parg_d':
#                             semEmt_trg_final = usp_canonical_form
#                         else:
#                             semEmt_trg_final = canonical_form
                    if str(preTermCanon) in ccontCanon2intSubgrEdges2cnt:
#                         print (bs, preTermCanon)
#                         print ("OK!")
                        if bs in bitVec2ccontCanon:
                            assert bitVec2ccontCanon[bs] == (preTermCanon, new_extEdge, new_extEdges_dicts,
                                                            trgs, dmrsNode2scope[node], 0)
                        else:
                            bitVec2ccontCanon[bs] = (preTermCanon, new_extEdge, new_extEdges_dicts,
                                                    trgs, dmrsNode2scope[node], 0)
                    else: #(exclusive?)
                        bitVec2newPsvItems[bs][(preTermCanon, new_extEdge)] = (new_extEdges_dicts, trgs,
                                                                                dmrsNode2scope[node], 0)
                    add_surf_prop = None
                    if dmrs_nxDG_ext.nodes[node]['instance'] == 'pron':
                        add_surf_prop = "obj"
                        for src, targ, lbl in dmrs_nxDG_orig.in_edges(node, data = 'label'):
                            if lbl == 'ARG1/NEQ':
                                for src2, targ2, lbl2 in dmrs_nxDG_orig.in_edges(src, data = 'label'):
                                    if dmrs_nxDG_orig.nodes[src2]['instance'] == "parg_d" and lbl2 == "ARG1/EQ":
                                        break
                                else:
                                    add_surf_prop = "subj"
                                    break
                            if lbl in ['ARG2/NEQ', 'ARG3/NEQ'] and dmrs_nxDG_orig.nodes[src]['instance'] == 'parg_d':
                                add_surf_prop = "subj"
                                break
                        
                    elif cop_trg_final:
                        if "tn:PR" in cop_trg_final or "tn:PA" in cop_trg_final and "pg:-" in cop_trg_final\
                            and "pf:-" in cop_trg_final and "sf:P" in cop_trg_final and "psv:ACT" in cop_trg_final\
                            and "neg:+" in cop_trg_final:
                            for src, targ, lbl in dmrs_nxDG_orig.out_edges(node, data = 'label'):
                                if "ARG1" in lbl:
    #                                 print (cop_trg_final)
                                    if  "tn:PR" in cop_trg_final:
                                        if str(dmrs_nxDG_orig.nodes[targ].get('pers')) == '3'\
                                            and dmrs_nxDG_orig.nodes[targ].get('num') == 'SG':
                                            add_surf_prop = "ps:" + str(dmrs_nxDG_orig.nodes[targ].get('pers'))\
                                                + ";nm:" + dmrs_nxDG_orig.nodes[targ].get('num') + "tn:PR"
                                            break
                                    elif "tn:PA" in cop_trg_final:
                                        if str(dmrs_nxDG_orig.nodes[targ].get('pers')) == '3'\
                                            and dmrs_nxDG_orig.nodes[targ].get('num') == 'SG':
                                            add_surf_prop = "ps:" + str(dmrs_nxDG_orig.nodes[targ].get('pers'))\
                                                + ";nm:" + dmrs_nxDG_orig.nodes[targ].get('num') + ";tn:PA"
                                            break
                        else:
                            add_surf_prop = cop_trg_final
                    bitVec2rule2oldItems[bs][preTermCanon].add(new_extEdge)
                    oldItem2histLogP[(bs, preTermCanon, new_extEdge)] =\
                        (((bs, (surface_canonical_form, add_surf_prop), "surface_canon"),), 0, None, new_extEdges_dicts,
                        trgs, dmrsNode2scope[node], 0)
#                     if bs == 2:
#                         print (bitVec2newPsvItems)
#                     bitVec2rule2oldItems2histLogP[bs][(preTermCanon, new_extEdge)].append(0)
#                     node2BitVec2items_new[node][bs] = bitVec2rule2oldItems2histLogP[bs]
#         print ()
#         for i in preTermCanonUsps2cnt:
#             print (i)
#             input()
        print ("number of nodes:", len(dmrs_nxDG_ext.nodes()))
        print ("starting with {} passive node sets".format(len(oldItem2histLogP)))
        
        # pprint (oldItem2histLogP)

        return

    global total_Ssucc_cnt, total_succ_cnt, times_spent, bleus_list, unrealized2cnt, sntId2log
    sntId2log[snt_id]['noOfNodes'] = len(dmrs_nxDG.nodes())
    sntId2log[snt_id]['original'] = true_sentence
    sntId2log[snt_id]['time'] = timeoutSeconds
    sntId2log[snt_id]['bleu'] = 0
    sntId2log[snt_id]['realized'] = None
    sntId2log[snt_id]['max_bleu'] = 0
    sntId2log[snt_id]['LPN-LPD-LRN-LRD'] = (0, 0, 0, 0)
    sntId2log[snt_id]['status'] = "Timeout"
    no_hyp = 0
    snt_succ = False
    Ssnt_succ = False
    merge_hist_set = None
#     node2BitVec2Items = defaultdict(defaultdict)
#     node2BitVec2items_new = defaultdict(defaultdict)
    bitVec2rule2oldItems = defaultdict(lambda: defaultdict(set))
    oldItem2histLogP = defaultdict()
    psvNodesBitVec2nbrsBitVec = defaultdict(int)
    bitVec2newPsvItems = defaultdict(defaultdict)
    bitVec2ccontCanon = defaultdict()
#     dmrs_node2nbrs = defaultdict(set)
    dmrs_nxDG_ext = None
    edgeKey2lbl = defaultdict()
    dmrs_node2bitVec = defaultdict()
    dmrsNode2scope = defaultdict()
    final_ltop = None
    pred2edgesBV2key2lbl = defaultdict(lambda: defaultdict(defaultdict))
    neg_edges = set()
    lexCanon2usp = defaultdict()
    node2pos = defaultdict()
    
    print (snt_id, true_sentence)
    # realize starts
#     print ("start--------------------------------")
    realized = None
    max_bleu, max_Srule, max_bleu_snt = 0, None, None
    targ_bleu, targ_Srule, realized = 0, None, None
    start = time.time()
    init_chart(dmrs_nxDG)
    replace_preterminals(snt_id, dmrs_nxDG, dmrs_nxDG_ext, dmrs_nxDG_ancAdj, true_deriv_nxDG, true_sentence, debug = True)
    # pprint (bitVec2newPsvItems)
    bitVec_complete = 2 ** len(dmrs_nxDG_ext.nodes()) -1

    Scompleted, completed, oldItem2histLogP, bitVec2rule2oldItems, ruleCheckCnts =\
        gen_hypothesis3(bitVec2rule2oldItems, oldItem2histLogP, dmrs_nxDG_ext, psvNodesBitVec2nbrsBitVec,
                               bitVec2newPsvItems, bitVec2ccontCanon, edgeKey2lbl, pred2edgesBV2key2lbl, dmrsNode2scope,
                               dmrs_node2bitVec, bitVec_complete, final_ltop, neg_edges, lexCanon2usp, node2pos, node_typed)

    end = time.time()
#     print ("end----------")
#     # realize ends
    print("time:", end - start)
    times_spent.append(end - start)
    print ("#hyp:", len(completed) + len(Scompleted))
    prec_numer, prec_denom, rec_numer, rec_denom = None, None, None, None
    if drawTree:
        util.write_figs_err(dmrs_nxDG, true_deriv_nxDG, true_sentence,
                            str(true_sentence).translate(str.maketrans('', '', string.punctuation))[:50] + "_orig")
    if Scompleted:
        sorted_Scompleted = sorted(Scompleted.items(), key = lambda x:x[1][1], reverse = True)
        for Scompleted in sorted_Scompleted:
            bitVec2predRule = defaultdict(list)
            toBeRealized = get_deriv(Scompleted[0], bitVec_complete, 1, None, "", build_tree = False,
                                     record_bitVecRule = not realized)
            realized_tmp, unrealized_list = get_realized(toBeRealized)
#             print (realized_tmp)
            bleu_score = score_with_bleu(realized_tmp, true_sentence)
            # print ("S", bleu_score.score, Scompleted[0][1], Scompleted[1][1], realized_tmp)
            if bleu_score.score > max_bleu:
                max_bleu, max_Srule, max_bleu_snt = bleu_score.score, Scompleted[0][1], realized_tmp
            if not realized:
                targ_bleu = bleu_score.score
                targ_Srule = Scompleted[0][1]
                realized = realized_tmp
                if unrealized_list:
                    print ("unrealized: ", unrealized_list)
                    unrealized2cnt += Counter(unrealized_list)
                if drawTree:
                    restored_deriv = nx.DiGraph()
                    get_deriv(Scompleted[0], bitVec_complete, 1, None, "", build_tree = True)
                    util.write_figs_err(None, restored_deriv, true_sentence,
                                        str(true_sentence).translate(str.maketrans('', '', string.punctuation))[:50])
                prec_numer, prec_denom, rec_numer, rec_denom = eval_derivSim.get_deriv_sim(true_deriv_nxDG, bitVec2predRule,
                                                                             dmrs_nxDG_ext, dmrs_node2bitVec)
                if prec_denom > 0 and rec_denom > 0:
                    print ("LP:", prec_numer/prec_denom, "LR:", rec_numer/rec_denom)
            Ssnt_succ = True
#             break
    if completed:
        print ("---")
        sorted_completed = sorted(completed.items(), key = lambda x:x[1][1], reverse = True)
        for completed in sorted_completed:
            bitVec2predRule = defaultdict(list)
            if drawTree:
                restored_deriv = nx.DiGraph()
                get_deriv(completed[0], bitVec_complete, 1, None, "", build_tree = True)
                # util.write_figs_err(None, restored_deriv, true_sentence,
                                    # str(completed[0][1]).translate(str.maketrans('', '', string.punctuation))[:50])
            toBeRealized = get_deriv(completed[0], bitVec_complete, 1, None, "", build_tree = False,
                                     record_bitVecRule = not realized)
            realized_tmp, unrealized_list = get_realized(toBeRealized)
#             print (realized_tmp)
#             total_succ_cnt += 1
            bleu_score = score_with_bleu(realized_tmp, true_sentence)
            # print ("NS", bleu_score.score, completed[0][1], completed[1][1], realized_tmp)
            if bleu_score.score > max_bleu:
                max_bleu, max_Srule, max_bleu_snt = bleu_score.score, completed[0][1], realized_tmp
            if not realized:
                targ_bleu = bleu_score.score
                targ_Srule = completed[0][1]
                realized = realized_tmp
                if unrealized_list:
                    print ("unrealized: ", unrealized_list)
                    unrealized2cnt += Counter(unrealized_list)
                prec_numer, prec_denom, rec_numer, rec_denom = eval_derivSim.get_deriv_sim(true_deriv_nxDG, bitVec2predRule,
                                                                             dmrs_nxDG_ext, dmrs_node2bitVec)
                if prec_denom > 0 and rec_denom > 0:
                    print ("LP:", prec_numer/prec_denom, "LR:", rec_numer/rec_denom)
            snt_succ = True
#             break

    if not snt_succ and not Ssnt_succ:
        # fallback
        print ("XXX")
        bitVec_covered = 0
        realized = ""
        bitVec_fbs = sorted([(bv, bin(bv).count("1")) for bv in list(bitVec2rule2oldItems.keys())], key = lambda t:t[1],
                            reverse = True)
        for idx, (bitVec_fb, cnt1) in enumerate(bitVec_fbs):
#             if idx > 150: break
            if bitVec_covered & bitVec_fb != 0: continue
            bitVec_covered = bitVec_covered | bitVec_fb
            if bitVec_covered == bitVec_complete: break
    #         print (bitVec2rule2oldItems)
            completed_fb = max([(bitVec_fb, rule, extEdge) 
                for rule, extEdges in bitVec2rule2oldItems[bitVec_fb].items()
                    for extEdge in extEdges], key = lambda t: oldItem2histLogP[t][1])
        
            toBeRealized = get_deriv(completed_fb, bitVec_fb, 1, None, "", build_tree = False)
        
            # restored_deriv = nx.DiGraph()
            # toBeRealized = get_deriv(completed_fb, bitVec_fb, 1, None, "", build_tree = True)
            # util.write_figs_err(None, restored_deriv, true_sentence,
                                       # str(completed_fb) + str(true_sentence).translate(str.maketrans('', '', string.punctuation))[:50])
            
            realized_tmp, unrealized_list = get_realized(toBeRealized)
            print (realized_tmp)
            if realized_tmp != "":
                realized += realized_tmp[0].lower() + realized_tmp[1:-1] + " "
#             if unrealized_list:
#                 print (unrealized_list)
#                 unrealized2cnt += Counter(unrealized_list)
        realized = realized.strip() + "."
        print (realized)
#         print (bin(bitVec_covered))
#         print (bin(bitVec_complete))
        bleu_score = score_with_bleu(realized, true_sentence)
        print (bleu_score.score)
        if not targ_bleu: targ_bleu = bleu_score.score
        if bleu_score.score > max_bleu:
            max_bleu, max_Srule, max_bleu_snt = bleu_score.score, None, realized
        
    if snt_succ or Ssnt_succ:
        total_succ_cnt += 1
        bleus_list.append(targ_bleu)
    if Ssnt_succ: total_Ssucc_cnt += 1
    print ("*****************************")
    if max_bleu_snt:
        print ("max:", max_bleu, "|", max_Srule, max_bleu_snt)
        
    print ("chosen:", targ_bleu, "|", targ_Srule, realized)
    print (snt_id, true_sentence)
    print ("{}/{}; {}/{}; {}/{}; {}/{}|| {}/{}; {}/{}".format(*ruleCheckCnts))
    print ("=========================================")
    
    sntId2log[snt_id]['time'] = end - start
    sntId2log[snt_id]['bleu'] = targ_bleu
    sntId2log[snt_id]['realized'] = realized
    sntId2log[snt_id]['max_bleu'] = max_bleu
    sntId2log[snt_id]['LPN-LPD-LRN-LRD'] = (prec_numer, prec_denom, rec_numer, rec_denom)
    if Ssnt_succ: sntId2log[snt_id]['status'] = "S"
    elif snt_succ: sntId2log[snt_id]['status'] = "Not S"
    else: sntId2log[snt_id]['status'] = "Partial"
        
#     if completed:
#         print ("result: >=1 parse found!!")
#     else: print ("result: no parse")
#     print ("==========") 
#     print (no_hyp)
# #     pprint (merge_hist_set)
    return realized, snt_succ or Ssnt_succ, (prec_numer, prec_denom, rec_numer, rec_denom)
    

def iterate_realize(test_sntid2item, pred_filepath, gs_filepath, predBlk_filepath, gsBlk_filepath, evaluate = False, debug = True, wsjOnly = False, brownOnly = False, wikiOnly = False, testSemEmt = None, testOn = None, testSntId = None, drawTree = False):
    node_cnt = Counter()
    canon_hyp_cnt = Counter()
    global total_Ssucc_cnt, total_succ_cnt, bleus_list, times_spent, processed_cnt
    processed_cnt = 0
    bleus_list = []
    times_spent = []
    total_precRec = [0,0,0,0]
    total_succ_cnt, total_Ssucc_cnt = 0, 0
    if evaluate:
        with open(predBlk_filepath, 'w'):
            pass
        with open(gsBlk_filepath, 'w'):
            pass
        with open(pred_filepath, 'w'):
            pass
        with open(gs_filepath, 'w'):
            pass
    for idx, snt_id in enumerate(tqdm(test_sntid2item)):
        if idx != 472: #not in [1,3,4,5]:
            pass
        if testOn != None:
            if idx != testOn: continue
        if wsjOnly:
            if snt_id[:3] != "wsj": continue
        elif brownOnly:
            if snt_id[:2] not in ["cf", "cg", "ck", "cl", "cm", "cn", "cp", "cr"]: continue
        elif wikiOnly:
            if snt_id[:2] not in ["ws"] or snt_id[:3] == "wsj": continue
        if testSntId and snt_id != testSntId: # wsj18c/21853022
            continue
#             continue
        dmrs_nodelink_dict = test_sntid2item[snt_id]["dmrs_nodelinkdict"]
        dmrs_nxDG = nx.readwrite.json_graph.node_link_graph(dmrs_nodelink_dict)
#         if len(dmrs_nxDG.nodes) >= 35:
#             continue
        dmrs_ancAdj_nodelink_dict = test_sntid2item[snt_id]["ancAdj_dmrs_nodelinkdict"]
        dmrs_nxDG_ancAdj = nx.readwrite.json_graph.node_link_graph(dmrs_ancAdj_nodelink_dict)
        anno_deriv_nxDG = None
        sentence = test_sntid2item[snt_id]["sentence"]
        if sentence not in ['Here is a list of the source and target languages SYSTRAN works with.',
                            'The magazine had its best year yet in 1988, when it celebrated its centennial and racked up a 17% gain in ad pages, to 283.']:
            pass
        # "By no means are these isolated cases.", "Also of interest are the mountains around the Rosendal community in Kvinnherad kommune.", "The giant waves are more dangerous on flat shores than on steep ones."
        # if sentence not in ["Here is a list of the source and target languages SYSTRAN works with."]:
        #    continue
        if debug or testSemEmt in [True, False]:
            anno_derivation_nodelink_dict = test_sntid2item[snt_id]["anno_derivation_nodelinkdict"]
            anno_deriv_nxDG = nx.readwrite.json_graph.node_link_graph(anno_derivation_nodelink_dict)
#             util.write_figs_err(dmrs_nxDG, anno_deriv_nxDG, "sentence", "abc" + "_orig")
#             input()
            _, semEmts = train_lex.align_dmrs_to_annoderiv(snt_id, dmrs_nxDG, anno_deriv_nxDG, sentence) 
            if testSemEmt == True and not semEmts or testSemEmt == False and semEmts:
                continue
            # print (sentence, semEmts, testSemEmt == True and not semEmts)
            # input()
        print (idx)
        
        # Start the timer. Once 5 seconds are over, a SIGALRM signal is sent.
        if not hookCheck:
            signal.alarm(timeoutSeconds)    
        # This try/except loop ensures that 
        #   you'll catch TimeoutException when it's sent.
        realized, covered, PrecRec = None, None, None
        try:
            realized, covered, PrecRec = realize_dmrs2(snt_id, dmrs_nxDG, dmrs_nxDG_ancAdj, anno_deriv_nxDG, sentence, debug = debug, drawTree = drawTree) # Whatever your function that might hang
            processed_cnt += 1
            if covered:
                total_precRec = list(map(add, total_precRec, PrecRec))
            if evaluate:
                with open(predBlk_filepath, 'a') as f:
                    f.write(realized + "\n")
                with open(gsBlk_filepath, 'a') as f:
                    f.write(sentence + "\n")
                if covered:
                    with open(pred_filepath, 'a') as f:
                        f.write(realized + "\n")
                    with open(gs_filepath, 'a') as f:
                        f.write(sentence + "\n")
    #             traceback.print_exc()
    #             print (snt_id, sentence, "failed")
    #         canon_hyp_cnt[len(hypotheses)] += 1
        except timeout.TimeoutException:
            print ("time exceeded {}s".format(timeoutSeconds))
            sntId2log[snt_id]['time'] = timeoutSeconds
            sntId2log[snt_id]['status'] = "Timeout"
            continue # continue the for loop if function A takes more than 5 second
        else:
            # Reset the alarm
            if not hookCheck:
                signal.alarm(0)
        
    print ("Finally: {}({}) out of {} got parse".format(total_succ_cnt, total_Ssucc_cnt, processed_cnt))
    print ("Time- max: {}, min: {}, avg: {}".format(max(times_spent), min(times_spent), sum(times_spent)/len(times_spent)))
    if bleus_list:
        print ("BLEU- max: {}, min: {}, avg: {}".format(max(bleus_list), min(bleus_list), sum(bleus_list)/len(bleus_list)))
#         print (bleus_list)
    print ("LPN, LPD, LRN, LRD:", total_precRec)
    lpn, lpd, lrn, lrd = total_precRec
    p, r = lpn/lpd, lrn/lrd
    print ("LP, LR, F1:", p, r, 2*p*r/(p+r))
    
    return unrealized2cnt, total_precRec


def main2(annotatedData_dir, extracted_rules_dir, redwdOrGw, config_name, sampleOrAll, debug = True, lexicalize = False, wsjOnly = False, brownOnly = False, wikiOnly = False, testSemEmt = None,
          use_phrg = False, nodeTyped = False, force_semEmt = False, no_semEmt = False, hook_check = True, evaluate = False, testOn = None, testSntId = None, drawTree = False):
    
    global rule2commaProb, startRule2cnt, ergRule2cnt, dmrsSubgrCanon2bTag2cnt, preTermCanon2bTagSurface2cnt, preTerm2surface2cnt, preTermAnno2surface2cnt, preTermCanon2surface2cnt, preTermCanonUsps2cnt, dgtrs2unaryRule2cnt, dgtrs2binaryRule2cnt, PHRG, SHRG, SHRG_coarse, ccontCanon2intSubgrEdges2cnt, dgtrsTrgs2edges2surf2logProb, canon2cnt, canon_usp2cnt, dgtrs2unaryRule2logProb, dgtrs2binaryRule2logProb, eqAnc_semiCanons, eqAnc_semiCanons_usp, ccont_semiCanons
    global max_subgraph_size, lexicalized, usePHRG, node_typed, forceSemEmt, noSemEmt, hookCheck, min_preTermCanon_freq, startSymRstr
    global exactCanonsToMatch, ccontCanonsToMatch, uspCanonsToMatch 
    
    sys.stderr.write("Loading config file ...\n")
    config, config_filepath = util.load_config(config_name, redwdOrGw)
    
    max_subgraph_size = config['train']['max_subgraph_size']
    lexicalized = lexicalize
    usePHRG = use_phrg
    node_typed = nodeTyped
    forceSemEmt = force_semEmt
    noSemEmt = no_semEmt
    hookCheck = hook_check
    #if redwdOrGw in ['gigaword', 'combined']:
    #    startSymRstr = False
#     min_preTermCanon_freq = config['test']['min_preTermCanon_freq']
    min_preTermCanon_freq = 0
    
    extracted_file2data = get_extractedFileName2data()
    
    
    # pprint (config
     
    all_training_data_filepath = os.path.join(annotatedData_dir, "training", "training_sntid2item.json")
    all_dev_data_filepath = os.path.join(annotatedData_dir, "dev", "dev_sntid2item.json")
    all_test_data_filepath = os.path.join(annotatedData_dir, "test", "test_sntid2item.json")
    sample_training_data_filepath = os.path.join(annotatedData_dir, "training", "sample_training_sntid2item.json")
    sample_dev_data_filepath = os.path.join(annotatedData_dir, "dev", "sample_dev_sntid2item.json")
    sample_test_data_filepath = sample_training_data_filepath
    
    if sampleOrAll == 'sample':
        training_data_filepath = sample_training_data_filepath
        dev_data_filepath = sample_dev_data_filepath
        test_data_filepath = sample_test_data_filepath
    elif sampleOrAll == 'all':
        training_data_filepath = all_training_data_filepath
        dev_data_filepath = all_dev_data_filepath
        test_data_filepath = all_test_data_filepath

    sys.stderr.write("Loading annotated test data ...\n")
    with open(test_data_filepath, "r", encoding='utf-8') as f:
        test_data = json.load(f)
            
    sys.stderr.write("Loaded!\n")
    
    extracted_rules_dir += "-28102021"
    extracted_rules_conf_dir = os.path.join(extracted_rules_dir, redwdOrGw + '-'\
                                    + config_name)
    if wsjOnly or brownOnly or wikiOnly:
        extracted_rules_conf_dir += '-wsj'
    if lexicalize:
        extracted_rules_conf_dir += "-lex"
    if nodeTyped:
        extracted_rules_conf_dir += "-nodeTyped"
    if testOn != None:
        extracted_rules_conf_dir += '-test'
    extracted_rules_confSA_dir = os.path.join(extracted_rules_conf_dir, 'all')
#     os.makedirs(extracted_rules_confSA_dir, exist_ok=True)
    
    sys.stderr.write('Start loading rules and PCFG from rule files!\n')
#     if not extracted_file2data["ergRule2cnt"]:
    for name in extracted_file2data:
        extracted_rules_confSA_path = os.path.join(extracted_rules_confSA_dir, name + ".json")
        with open(extracted_rules_confSA_path, "r") as f:
            extracted_file2data[name] = json.load(f)
    sys.stderr.write('Done!\n')
    
    startRule2cnt, ergRule2cnt, canon2cnt, canon_usp2cnt = [Counter(extracted_file2data[key]) for key in "startRule2cnt, ergRule2cnt, canon2cnt, canon_usp2cnt".split(", ")]
    eqAnc_semiCanons, eqAnc_semiCanons_usp, ccont_semiCanons = [set(extracted_file2data[key]) for key in "eqAnc_semiCanons, eqAnc_semiCanons_usp, ccont_semiCanons".split(", ")]
    
    ccont_semiCanons_usp = set()
    eqAnc_semiCanons = eqAnc_semiCanons.union(ccont_semiCanons)
    eqAnc_semiCanons_usp = eqAnc_semiCanons_usp.union(ccont_semiCanons)
    for key_str in eqAnc_semiCanons:
        key = make_tuple(key_str)
#         print (key)
#         ccont_cnter = Counter(ccont_key)
        key_usp = util.get_semicanonical_form(key, underspecLemma = True, from_exact_semi = True)
        eqAnc_semiCanons_usp.add(key_usp)
#         print (key_usp)
#     for k in eqAnc_semiCanons_usp:
#         pprint (k)
#     input()
    
    rule2commaProb, PHRG, SHRG, SHRG_coarse, dgtrsTrgs2edges2surf2logProb, dmrsSubgrCanon2bTag2cnt, preTermCanon2bTagSurface2cnt, preTerm2surface2cnt, preTermAnno2surface2cnt, preTermCanon2surface2cnt, preTermCanonUsps2cnt, dgtrs2unaryRule2cnt, dgtrs2binaryRule2cnt, ccontCanon2intSubgrEdges2cnt, dgtrs2unaryRule2logProb, dgtrs2binaryRule2logProb = [extracted_file2data[key] for key in "rule2commaProb, PHRG, SHRG, SHRG_coarse, dgtrsTrgs2edges2surf2logProb, dmrsSubgrCanon2bTag2cnt, preTermCanon2bTagSurface2cnt, preTerm2surface2cnt, preTermAnno2surface2cnt, preTermCanon2surface2cnt, preTermCanonUsps2cnt, dgtrs2unaryRule2cnt, dgtrs2binaryRule2cnt, ccontCanon2intSubgrEdges2cnt, dgtrs2unaryRule2logProb, dgtrs2binaryRule2logProb".split(", ")]

    # Filter canon to be used
    filter_preTermCanonUsp(preTermCanonUsps2cnt, min_preTermCanon_freq)
    SHRG_tmp = SHRG.copy()
#     pprint (SHRG_tmp)
    PHRG_tmp = PHRG.copy()
    SHRG = defaultdict(Counter)
    PHRG = defaultdict(defaultdict)
    global SHRG_rules2edges, SHRG_unaryTC, dgtrs2unaryRuleTC2logProb, dgtrPar2unaryChain, semEmtDgtrsRule, pred2dgtrs, predTrg2edge2surf2logProb
    for key, val in SHRG_tmp.items():
        shrg_key_tuple = make_tuple(key)
        derivRules, intSubgrEdges = shrg_key_tuple
        intSubgrEdges_sets = None
        if intSubgrEdges:
            intSubgrEdges_sets = tuple(frozenset(es) for es in intSubgrEdges)
#             print (intSubgrEdges_sets)
        SHRG[(derivRules, intSubgrEdges_sets)] = val
        SHRG_rules2edges[derivRules].append(intSubgrEdges_sets)
    for key, val in PHRG_tmp.items():
        phrg_key_tuple = make_tuple(key)
        derivRules, intSubgrEdges = phrg_key_tuple
        intSubgrEdges_sets = None
        if intSubgrEdges:
            intSubgrEdges_sets = tuple(frozenset(es) for es in intSubgrEdges)
        PHRG[(derivRules, intSubgrEdges_sets)] = val
        
    
    dmrsSubgrCanon2bTag2cnt_tmp = dmrsSubgrCanon2bTag2cnt.copy()
    dmrsSubgrCanon2bTag2cnt = defaultdict(Counter)
    for canon, btag2cnt in dmrsSubgrCanon2bTag2cnt_tmp.items():
        key = make_tuple(canon)
        dmrsSubgrCanon2bTag2cnt[key] = Counter(btag2cnt)
    
    dgtrs2binaryRule2logProb_tmp = dgtrs2binaryRule2logProb.copy()
    dgtrs2binaryRule2logProb = defaultdict(defaultdict)
    for dgtrs, rule2logProb in dgtrs2binaryRule2logProb_tmp.items():
        for rule, logProb in rule2logProb.items():
            dgtrs_t = make_tuple(dgtrs)
            dgtrs2binaryRule2logProb[dgtrs_t][rule] = logProb
#             print (dgtrs_t)f
#             if "hdn_optcmp_c&N" in dgtrs_t:
#                 print (dgtrs)
    dgtrs2unaryRule2logProb_tmp = dgtrs2unaryRule2logProb.copy()
    dgtrs2unaryRule2logProb = defaultdict(defaultdict)
    for dgtrs, rule2logProb in dgtrs2unaryRule2logProb_tmp.items():
        for rule, logProb in rule2logProb.items():
            try:
                dgtrs_t = make_tuple(dgtrs)
            except:
                dgtrs_t = dgtrs
#             if dgtrs_t == "hdn_bnp-num_c&NP":
#                 print (rule)
            dgtrs2unaryRule2logProb[dgtrs_t][rule] = logProb
    
    pred2dgtrs = defaultdict(set)
    semEmtDgtrsRule = defaultdict(set)
    predTrg2edge2surf2logProb = defaultdict(lambda: defaultdict(lambda: defaultdict(defaultdict)))
    dgtrsTrgs2edges2surf2logProb_tmp = dgtrsTrgs2edges2surf2logProb.copy()
    dgtrsTrgs2edges2surf2logProb = defaultdict(lambda: defaultdict(lambda: defaultdict(defaultdict)))
    for semEmt_type, dgtrsTrgs2edges2surf2logProb_semEmt_tmp in dgtrsTrgs2edges2surf2logProb_tmp.items():
        for dgtrsTrg in dgtrsTrgs2edges2surf2logProb_semEmt_tmp:
            dgtrsTrgs = make_tuple(dgtrsTrg)
#             print (dgtrsTrgs)
            if semEmt_type != "predSemEmt":
                semEmtDgtrsRule[semEmt_type].add((dgtrsTrgs[0][0], dgtrsTrgs[1][0]))
            else:
                semEmtDgtrsRule[semEmt_type].add(dgtrsTrgs[0])
                predTrg = dgtrsTrgs[1]
#                 print (dgtrsTrgs[0], predTrg)
            for edge in dgtrsTrgs2edges2surf2logProb_semEmt_tmp[dgtrsTrg]:
#                 if semEmt_type != "predSemEmt":
                edges_set = tuple(frozenset(edges) for edges in make_tuple(edge))
#                 else:
#                     edges_set = edge
                for derivRuleSurf, logProb in dgtrsTrgs2edges2surf2logProb_semEmt_tmp[dgtrsTrg][edge].items():
                    derivRule, surf = make_tuple(derivRuleSurf)
                    if surf == ('<left>', '<right>'):
    #                     print (dgtrsTrg, "?")
                        pass
                    else:
                        if semEmt_type == "predSemEmt":
                            dgtrsTrgs2edges2surf2logProb[semEmt_type][dgtrsTrgs[0]][edges_set][(derivRule, surf, predTrg)] = logProb
                            predTrg2edge2surf2logProb[predTrg[0]][predTrg[1]][(derivRule, surf)] = logProb
                            pred2dgtrs[predTrg[0]].add((dgtrsTrgs[0]))
                        else:
                            dgtrsTrgs2edges2surf2logProb[semEmt_type][dgtrsTrgs][edges_set][(derivRule, surf)] = logProb
#                             if semEmt_type == 'copula' and dgtrsTrgs[0][0] == 'hdn_bnp_c&NP' and dgtrsTrgs[1][0] == 'hd_optcmp_c&VP':
#                                 pass
#                                 print (dgtrsTrgs)
#                                 print (edges_set)
#                                 print (derivRule)
#                                 print (surf)
#                                 print ("===========")
#     print (pred2dgtrs)
#     pprint (predTrg2edge2surf2logProb)
#     print (len(dgtrsTrgs2edges2surf2logProb_copula))
    
    # get transitive closures
    SHRG_unary = defaultdict(set)
    for key in SHRG_rules2edges:
        if len(key) > 1:
            continue
        for derivRule_new in SHRG[(key, None)]:
            SHRG_unary[key].add(derivRule_new)
#             input()
    SHRG_unaryTC = copy.deepcopy(SHRG_unary)
    SHRG_unaryTC_tmp = copy.deepcopy(SHRG_unaryTC)
    dgtrs2unaryRuleTC2logProb = copy.deepcopy(dgtrs2unaryRule2logProb)
    dgtrs2unaryRuleTC2logProb_tmp  = copy.deepcopy(dgtrs2unaryRule2logProb)
#     print (dgtrs2unaryRule2logProb.keys())
#     pprint (SHRG_unaryTC)
    for i in range(max_unary_chain-1):
        for key, key2s in SHRG_unaryTC_tmp.items():
            for key2 in key2s:
                if i == 0:
                    dgtrPar2unaryChain[(key[0], key2)] = [key[0], key2]
                for key3 in SHRG_unary[(key2,)]:
    #             if key2 in 
#                     print (key3)
                    if key3 != key[0]:
                        SHRG_unaryTC[key].add(key3)
#                         print (key, key3)
#                         input()
#         pprint (SHRG_unaryTC)
        SHRG_unaryTC_tmp = copy.deepcopy(SHRG_unaryTC)
        for key, key2s in dgtrs2unaryRuleTC2logProb_tmp.items():
#             print (key)
#             input()
            for key2 in key2s:
                if key2 not in dgtrs2unaryRule2logProb: continue
                for key3 in dgtrs2unaryRule2logProb[key2]:
                    if key3 == key: continue
                    new_logProb = dgtrs2unaryRuleTC2logProb_tmp[key][key2]\
                        + dgtrs2unaryRule2logProb[key2][key3]
                    if key3 in dgtrs2unaryRuleTC2logProb[key]:
                        if new_logProb <= dgtrs2unaryRuleTC2logProb[key][key3]:
                            continue
                    dgtrs2unaryRuleTC2logProb[key][key3] = new_logProb
                    dgtrPar2unaryChain[(key, key3)] = dgtrPar2unaryChain[(key, key2)][:-1] + dgtrPar2unaryChain[(key2, key3)]
                        
#                             print (key, key2, key3)
        dgtrs2unaryRuleTC2logProb_tmp = copy.deepcopy(dgtrs2unaryRuleTC2logProb)
    # pprint (dgtrPar2unaryChain)
    r2edgesCnt = Counter()
    typesOfEdges = Counter()
    for rule, edges in SHRG_rules2edges.items():
        r2edgesCnt[len(edges)] += 1
        for edge_set in edges:
            if edge_set:
                for edges in edge_set:
                    for edge in edges:
                        typesOfEdges[edge] += 1
#     pprint (r2edgesCnt)
#     print (len(typesOfEdges))
#     pprint (typesOfEdges)
#     input()
    global ccont_preds
    for t in ccont_semiCanons:
        for pred, pred_cnt in make_tuple(t):
            ccont_preds.add(pred)
    
    ccontCanonsToMatch = set(make_tuple(s) for s in set(ccontCanon2intSubgrEdges2cnt.keys()))
    exactCanonsToMatch = set(make_tuple(s)
                                 for s in set(preTermCanon2bTagSurface2cnt.keys()).union(
                                     set(preTermCanon2surface2cnt.keys()))
                             ) 
#     for can in ccontCanonsToMatch:
#         if "unknown" in str(can) and "implicit" in str(can):
#             print (can)
#             print ()
    uspCanonsToMatch = set(make_tuple(s) for s in list(preTermCanonUsps2cnt.keys()))
    
    preTermCanon2surface2cnt_tmp = preTermCanon2surface2cnt
    preTermCanon2surface2cnt = defaultdict(Counter)
    for canon, surface2cnt in preTermCanon2surface2cnt_tmp.items():
        preTermCanon2surface2cnt[make_tuple(canon)] = Counter(surface2cnt)
    preTermCanon2bTagSurface2cnt_tmp = preTermCanon2bTagSurface2cnt
    preTermCanon2bTagSurface2cnt = defaultdict(Counter)
    for canon, surface2cnt in preTermCanon2bTagSurface2cnt_tmp.items():
        preTermCanon2bTagSurface2cnt[make_tuple(canon)] = Counter(surface2cnt)
    
        
        
    preTerm2surface2cnt_tmp = preTerm2surface2cnt
    preTerm2surface2cnt = defaultdict(Counter)
    preTermAnno2surface2cnt_tmp = preTermAnno2surface2cnt
    preTermAnno2surface2cnt = defaultdict(Counter)
    for preterm, surface2cnt in preTerm2surface2cnt_tmp.items():
        preTerm2surface2cnt[preterm] = Counter(surface2cnt)
    for pretermAnno, surface2cnt in preTermAnno2surface2cnt_tmp.items():
        preTermAnno2surface2cnt[pretermAnno] = Counter(surface2cnt)
    # |S cond. prob
    global SRule2logProb
#     S_denom = sum([v for k, v in ergRule2cnt.items() if k.endswith("&S")])
#     SRule2logProb = {k: math.log(v/S_denom) for k, v in ergRule2cnt.items() if k.endswith("&S")}
    S_denom = sum([v for k, v in startRule2cnt.items()])
    SRule2logProb = {k: math.log(v/S_denom) for k, v in startRule2cnt.items()}
        
#     for dgtrs in dgtrs2binaryRule2cnt:
#         d1, d2 = make_tuple(dgtrs)
#         if d1 in ["be_c_am", "to_c_prop"]:
#             print (dgtrs, dgtrs2binaryRule2cnt[dgtrs])
#             input()
#     ncnt = 0
#     nncnt = 0
#     for key in SHRG:
#         nodes, _ = key
#         if len(nodes) == 3:
#             if None in nodes:
# #                 print (key, SHRG[key])
#                 ncnt += 1
#                 BPsemEmtSHRGkeys.add(nodes)
#             else:
#                 nncnt += 1
#     print (ncnt, nncnt)
#     input()
#     for key in SHRG:
#         pprint (key)
#         input()
#     (('hdn_bnp-pn_c&NP', 'hd_optcmp_c&VP'), ((), ('#0-src-ARG1/NEQ-targ-#0', '#0-src-ARG1/NEQ-targ-#0')))
# (('hd_optcmp_c&VP', 'hdn_bnp-pn_c&NP'), (('#0-src-ARG1/NEQ-targ-#0', '#0-src-ARG1/NEQ-targ-#0'), ()))
    pcfg_str = ""
    nodeTyped_str = ""
    hookCheck_str = ""
    noSemEmt_str = ""
    testSemEmt_str = ""
    if not use_phrg:
        pcfg_str = '-pcfg'
    if node_typed:
        nodeTyped_str = "-nodeTyped"
    if not hookCheck:
        hookCheck_str = '-noHookCheck'
    if noSemEmt:
        noSemEmt_str = '-noSemEmt'
    if testSemEmt:
        testSemEmt_str = "-testOnSemEmt"
    elif testSemEmt == False:
        testSemEmt_str = "-testOnNoSemEmt"
    if wsjOnly:
        result_dir = os.path.join("results_{}".format(results_dir_suffix), redwdOrGw + '-' + config_name + noSemEmt_str + hookCheck_str + pcfg_str + nodeTyped_str +"-wsj" + testSemEmt_str, sampleOrAll)
    elif brownOnly:
        result_dir = os.path.join("results_{}".format(results_dir_suffix), redwdOrGw + '-' + config_name + noSemEmt_str + hookCheck_str + pcfg_str + nodeTyped_str + "-brown" + testSemEmt_str, sampleOrAll)
    elif wikiOnly:
        result_dir = os.path.join("results_{}".format(results_dir_suffix), redwdOrGw + '-' + config_name + noSemEmt_str + hookCheck_str + pcfg_str + nodeTyped_str + "-wiki" + testSemEmt_str, sampleOrAll)
    else:
        result_dir = os.path.join("results_{}".format(results_dir_suffix), redwdOrGw + '-' + config_name + noSemEmt_str + hookCheck_str + pcfg_str + nodeTyped_str + testSemEmt_str, sampleOrAll)
    if forceSemEmt:
        result_dir += "-forceSemEmt"
    os.makedirs(result_dir, exist_ok=True)
    pred_filepath = os.path.join(result_dir, "pred.txt")
    gs_filepath = os.path.join(result_dir, "gs.txt")
    predBlk_filepath = os.path.join(result_dir, "pred_fullCov.txt")
    gsBlk_filepath = os.path.join(result_dir, "gs_fullCov.txt")
    score_filepath = os.path.join(result_dir, "score.txt")
    scoreBlk_filepath = os.path.join(result_dir, "score_fullCov.txt")
    log_filepath = os.path.join(result_dir, "log.txt")
    
    sys.stderr.write('Start reazlization!\n')
    unrealized2cnt, total_precRec = iterate_realize(test_data, pred_filepath, gs_filepath, predBlk_filepath, gsBlk_filepath,
                                                    evaluate, debug, wsjOnly, brownOnly, wikiOnly, testSemEmt, testOn, testSntId, drawTree)
    sys.stderr.write('Done reazlization!\n')
    
    if evaluate:
        lpn, lpd, lrn, lrd = total_precRec
        p, r = lpn/lpd, lrn/lrd
        with open(log_filepath, "w") as f:
            json.dump(sntId2log, f)
        sys.stderr.write('Score reazlization!\n')
        pred = []
        refs = []
        # only evaluate non-blank lines
        with open(pred_filepath) as f:
            line = f.readline().strip()
            while(line):
                pred.append(line)
                line = f.readline().strip()
        with open(gs_filepath) as f:
            line = f.readline().strip()
            while(line):
                refs.append(line)
                line = f.readline().strip()
        assert len(pred) == len(refs)
        refs = [refs]
        corpus_bleu = sacrebleu.corpus_bleu(pred, refs)
        with open(score_filepath, "w") as f:
            f.write("Coverage - {} out of {} got parse; {} with 'good' start symbol\n".format(total_succ_cnt, processed_cnt, total_Ssucc_cnt))
            f.write("Time - max: {}, min: {}, avg: {}\n".format(max(times_spent), min(times_spent),
                                                              sum(times_spent)/len(times_spent)))
            f.write("BLEU - max: {}, min: {}, avg: {}\n".format(max(bleus_list), min(bleus_list),
                                                             sum(bleus_list)/len(bleus_list)))
            f.write("Corpus BLEU - {}\n".format(corpus_bleu.__dict__))
            f.write("LPN, LPD, LRN, LRD: {}\n".format(str(total_precRec)))
            f.write("LP: {}; LR: {}; F1: {}\n".format(p, r, 2*p*r/(p+r)))
        sys.stderr.write('Scored!\n')
        
        pred_fullCov = []
        refs_fullCov = []
        # evaluate  all lines
        with open(predBlk_filepath) as f:
            line = f.readline().strip()
            while(line):
                pred_fullCov.append(line)
                line = f.readline().strip()
        with open(gsBlk_filepath) as f:
            line = f.readline().strip()
            while(line):
                refs_fullCov.append(line)
                line = f.readline().strip()
        assert len(pred_fullCov) == len(refs_fullCov)
        refs_fullCov = [refs_fullCov]
        corpus_bleu = sacrebleu.corpus_bleu(pred_fullCov, refs_fullCov)
        with open(scoreBlk_filepath, "w") as f:
            f.write("Coverage - {} out of {} got parse; {} with 'good' start symbol\n".format(total_succ_cnt, processed_cnt, total_Ssucc_cnt))
            f.write("Time - max: {}, min: {}, avg: {}\n".format(max(times_spent), min(times_spent),
                                                              sum(times_spent)/len(times_spent)))
            f.write("BLEU (not full Cov) - max: {}, min: {}, avg: {}\n".format(max(bleus_list), min(bleus_list),
                                                             sum(bleus_list)/len(bleus_list)))
            f.write("Corpus BLEU - {}\n".format(corpus_bleu.__dict__))
        sys.stderr.write('Scored!\n')
    return unrealized2cnt, sntId2log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotatedData_dir', help='path to preprocessed data directory')
    parser.add_argument('extracted_rules_dir', help='path to extracted rules directory')
    parser.add_argument('redwdOrGw', help='rule extracted from redwoods or gigaword or combined')
    parser.add_argument('config_name', help='rule extracted under standard, dense, ... config')
    parser.add_argument('sampleOrAll', help='test with sample data only or not')
    parser.add_argument('wsjOnly', help='test on wsj only using rules extracted from wsj training data only')
    parser.add_argument('brownOnly', help='test on brown only using rules extracted from wsj training data only')
    parser.add_argument('wikiOnly', help='test on wikipedia only using rules extracted from wsj training data only')
    parser.add_argument('testSemEmt', help='test on instances with semEmt. True->w/; False->w/o; None->all')
    parser.add_argument('use_phrg', help='use phrg')
    parser.add_argument('nodeTyped', help='pshrg rule nwith node typed')
    parser.add_argument('force_semEmt', help='add semantically empty words whenever possible')
    parser.add_argument('no_semEmt', help='do not insert semantically empty words')
    parser.add_argument('hook_check', help='perform INDEX and LTOP validation')
    parser.add_argument('evaluate', help='evaluate results?')
    parser.add_argument('startSymRstr', help='restrict start symbol to be those seen in training data')
    
    
     
    args = parser.parse_args()
    main2(args.annotatedData_dir, args.extracted_rules_dir, args.redwdOrGw, args.config_name, args.sampleOrAll, args.wsjOnly, args.brownOnly, args.wikiOnly, args.testSemEmt, args.use_phrg, args.nodeTyped, args.force_semEmt, args.no_semEmt, args.hook_check, args.evaluate, args.startSymRstr)
    # main(args.annotatedData_dir, args.extracted_rules_dir, args.config_filepath)
