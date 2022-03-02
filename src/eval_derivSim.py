import src.train_lex as train_lex 
import src.util as util

from pprint import pprint

from collections import defaultdict

def get_trueBitVecRule(true_deriv, dmrs_nxDG_ext, dmrs_node2bitVec):
    
    def _dfs_extract(anno_deriv_nxDG_uni, curr_node, dmrs_anchors2nodes, print_debug = False):
        
        nonlocal bitVec2rule, dmrs_anchors2bitVec
        
        for src, targ in anno_deriv_nxDG_uni.out_edges(nbunch = [curr_node]):
            if not anno_deriv_nxDG_uni.nodes[targ]['extracted']:
                _dfs_extract(anno_deriv_nxDG_uni, targ, dmrs_anchors2nodes)
        
        curr_derivRule = anno_deriv_nxDG_uni.nodes[curr_node]['entity']
        curr_deriv_anchor = (int(anno_deriv_nxDG_uni.nodes[curr_node]['anchor_from']),
                             int(anno_deriv_nxDG_uni.nodes[curr_node]['anchor_to']))
        dgtrs = util.get_deriv_leftRight_dgtrs(anno_deriv_nxDG_uni, curr_node)
        if not dgtrs: return
        if all([anno_deriv_nxDG_uni.nodes[targ]['extracted']
                    for targ in dgtrs]):
            if anno_deriv_nxDG_uni.nodes[curr_node]['bitVec']:
                anno_deriv_nxDG_uni.nodes[curr_node]['extracted'] = True
                return
            #daugther=2
            if len(dgtrs) == 2:
                ccont_bitVec = dmrs_anchors2bitVec.get(curr_deriv_anchor)
                if ccont_bitVec == None: ccont_bitVec = 0
                else:
                    pass
#                     print (ccont_bitVec, curr_derivRule)
                bitVecs = [anno_deriv_nxDG_uni.nodes[dgtr]['bitVec'] for dgtr in dgtrs]
                if bitVecs != [None, None]:
                    if not None in bitVecs:
                        assert ccont_bitVec & bitVecs[0] & bitVecs[1] == 0
#                         print (bitVecs)
                        bitVec_new = ccont_bitVec | bitVecs[0] | bitVecs[1]
                        anno_deriv_nxDG_uni.nodes[curr_node]['bitVec'] = bitVec_new
                        bitVec2rule[bitVec_new].append(curr_derivRule.split("&")[0].split("^")[0])
                    else:
#                         print (bitVecs, "N!")
                        anno_deriv_nxDG_uni.nodes[curr_node]['bitVec'] = bitVecs[0] or bitVecs[1]
            # daugther=1
            elif len(dgtrs) == 1:
                dgtr = dgtrs[0]
                bitVec = anno_deriv_nxDG_uni.nodes[dgtr]['bitVec']
                if bitVec:
#                     print (bitVec, "U")
                    anno_deriv_nxDG_uni.nodes[curr_node]['bitVec'] = bitVec
                    bitVec2rule[bitVec].append(curr_derivRule.split("&")[0].split("^")[0])
            anno_deriv_nxDG_uni.nodes[curr_node]['extracted'] = True
                
    bitVec2rule = defaultdict(list)
    annoDerivPreTerm2dmrsNodeSubgr, semEmtLexEnt \
        = train_lex.align_dmrs_to_annoderiv(None, dmrs_nxDG_ext, true_deriv, None)
    anno_deriv_nxDG_uni = true_deriv.copy()
    dmrs_anchors2bitVec = defaultdict()
    dmrs_anchors2nodes, dmrs_anchors2preds = util.get_equalanchor_nodes(dmrs_nxDG_ext, lexical_only = False)
    for anc, dmrs_nodes in dmrs_anchors2nodes.items():
        bitVec = 0
        for dmrs_node in dmrs_nodes:
            bitVec = bitVec | dmrs_node2bitVec[dmrs_node]
        dmrs_anchors2bitVec[anc] = bitVec
    
    for node, node_prop in anno_deriv_nxDG_uni.nodes(data = True):
        if not 'cat' in node_prop: continue
        anno_deriv_nxDG_uni.nodes[node]['bitVec'] = None
        if node_prop['cat'] != '<terminal>':
            anno_deriv_nxDG_uni.nodes[node]['extracted'] = False
        else:
            anno_deriv_nxDG_uni.nodes[node]['extracted'] = True
        if node in annoDerivPreTerm2dmrsNodeSubgr:
            bitVec = 0
            for dmrs_node in annoDerivPreTerm2dmrsNodeSubgr[node]:
                bitVec = bitVec | dmrs_node2bitVec[dmrs_node]
            anno_deriv_nxDG_uni.nodes[node]['bitVec'] = bitVec
#             print (bitVec)
        else:
            anno_deriv_nxDG_uni.nodes[node]['bitVec'] = None
#     util.write_figs_err(None, anno_deriv_nxDG_uni, None)
    _dfs_extract(anno_deriv_nxDG_uni, anno_deriv_nxDG_uni.graph['root'], dmrs_anchors2nodes)
#     util.write_figs_err(None, anno_deriv_nxDG_uni, None)
    return bitVec2rule

def get_deriv_sim(true_deriv, bitVec2predRule, dmrs_nxDG_ext, dmrs_node2bitVec):
    bitVec2rule = get_trueBitVecRule(true_deriv, dmrs_nxDG_ext, dmrs_node2bitVec)
#     pprint (bitVec2rule)
#     pprint (bitVec2predRule)
    prec_numer, prec_denom, rec_numer, rec_denom = 0, 0, 0, 0
    for bitVec, predRules in bitVec2predRule.items():
        predRules.reverse() #bottom-to-top
        prec_denom += 1
        if bitVec in bitVec2rule and predRules[0] == bitVec2rule.get(bitVec)[0]:
            prec_numer += 1
        # print (predRules, bitVec2rule.get(bitVec))
        # for predRule in predRules:
        #     prec_denom += 1
        #     if bitVec in bitVec2rule and predRule in bitVec2rule[bitVec]:
        #         prec_numer += 1
    # print ()
    for bitVec, trueRules in bitVec2rule.items():
        #trueRules
        rec_denom += 1
        if bitVec in bitVec2predRule and trueRules[0] == bitVec2predRule.get(bitVec)[0]:
            rec_numer += 1
        # print (trueRules, bitVec2predRule.get(bitVec))
        # for trueRule in trueRules:
        #     rec_denom += 1
        #     if bitVec in bitVec2predRule and trueRule in bitVec2predRule[bitVec]:
        #         rec_numer += 1
    return prec_numer, prec_denom, rec_numer, rec_denom
    