import re
from collections import defaultdict,Counter
from functools import cmp_to_key
try:
    __IPYTHON__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm
import pickle
import time

from networkx.drawing.nx_agraph import to_agraph
from networkx.algorithms.boundary import edge_boundary
from networkx.algorithms.components import is_weakly_connected

import networkx as nx

from pprint import pprint
import string
import traceback
import json

from src import dg_util, timeout

lexeme_pred = set(["abstr_deg","basic_card","card","cop_id","dofm","dofw","ellipsis_expl","ellipsis_ref",
                       "elliptical_n","excl","fraction","generic_entity","id","interval","little-few_a",
                       "manner","measure","mofy","much-many_a","named","named_n","ne_x","neg",
                       "numbered_hour","ord","part_of","person","person_n","place_n","polite",
                       "season","superl","temp","thing","time","time_n","year_range","yofc", # from dm.cfg
                  "free_relative_ever_q","recip_pro","comp_enough","comp_not+so","comp_not+too","comp_equal","comp_less","comp_so","comp_too","comp",
                   "poss","minute","all+too","few+if+any_a","generic_verb","greet","holiday","not_x_deg","some_q","ellipsis","meas_np","temp_loc_x","timezone_p","off_a_1","no_a_1","free_relative_q"]) # added by me; also only pron with pt!:ZERO or pers:1
lexical_mod_pred = set(['parg_d','comp','superl'])
structural_pred = set(["compound","focus_d","implicit_conj","unknown","appos","nominalization","subord",
                   "parenthetical","eventuality","with_p","relative_mod",
                      "loc_nonsp", "num_seq",
                       "addressee","discourse","fw_seq","idiom_q_i"]) # also pron with pt:ZERO and pers!:1
chunk_pred = set([])
particles = "to of for up on back out in than as about from with at off down into ahead away so over around along against aside through yet behind open across".split(" ")
copula = set("is 's am 'm are 're was were be been being do does did have 've has had having 'd will 'll to".split(" "))
relative_adverbs = set("when, where".split(", ")) # how, why handled by lexical preedicates
relative_pronouns = set("who, whom, that, what, which, whose".split(", ")) # whose handled (or not?) by poss as lexical predicate
complementizer = set()
utterance = set("umm, um, uh, ah, eh")
phraseToOmit = set("I must say, you know, fucking, kind of, let's say, that is, therefore, for instance, for example, I guess, basically, however, I think, maybe, say, I mean, that's right, though, perhaps, e.g., sort of, in a way, or so, sorry, for one, I dunno, I suppose, so to speak, alas, the hell, the devil, the fuck, the heck, oh where, on earth, in the world, exactly, else, more".lower().split(", "))
unknown2pos = {"jj": "a",
               "jjr": "a",
               "jjs": "a",
               "cd": "u",
               "nn": "n",
               "nns": "n",
               "nnps": "n",
               "nnp": "n",
               "fw": "n",
               "rb": "a",
               "vb": "v",
               "vbg": "v",
               "vbn": "v",
               "vbp": "v",
               "vbz": "v",
               "vbd": "v", "a": "a", "p": 'p', "q": "q", "n":"n", "v":"v", "x":"x", "c": "c"}

erg_rules = set("aj-hd_int-inv_c aj-hd_int-rel_c aj-hd_int_c aj-hd_scp-pr_c aj-hd_scp-xp_c aj-hd_scp_c aj-hdn_adjn_c aj-hdn_norm_c aj-np_frg_c aj-np_int-frg_c aj-pp_frg_c aj-r_frg_c cl-cl_crd-im_c cl-cl_crd-int-t_c cl-cl_crd-m_c cl-cl_crd-rc-t_c cl-cl_crd-t_c cl-cl_runon-cma_c cl-cl_runon_c cl-np_runon-prn_c cl-np_runon_c cl-rc_c cl_cnj-frg_c cl_cp-frg_c cl_np-wh_c cl_rc-fin-modgap_c cl_rc-fin-nwh_c cl_rc-inf-modgap_c cl_rc-inf-nwh-sb_c cl_rc-inf-nwh_c cl_rc-instr_c cl_rel-frg_c det_prt-nocmp_dlr det_prt-of-agr_dlr det_prt-of-nagr_dlr flr-hd_nwh-nc-np_c flr-hd_nwh-nc_c flr-hd_nwh_c flr-hd_rel-fin_c flr-hd_rel-inf_c flr-hd_wh-mc-sb_c flr-hd_wh-mc_c flr-hd_wh-nmc-fin_c flr-hd_wh-nmc-inf_c hd-aj_cmod-s_c hd-aj_cmod_c hd-aj_int-sl_c hd-aj_int-unsl_c hd-aj_scp-pr_c hd-aj_scp_c hd-aj_vmod-s_c hd-aj_vmod_c hd-cl_fr-rel_c hd-cmp_2_c hd-cmp_u_c hd-hd_rnr-nv_c hd-hd_rnr_c hd-pct_c hd_imp_c hd_inv-nwh_c hd_optcmp_c hd_xaj-crd-s_c hd_xaj-int-s_c hd_xaj-int-vp_c hd_xcmp_c hd_xsb-fin_c hd_yesno_c hdn-aj_rc-pr_c hdn-aj_rc_c hdn-aj_redrel-pr_c hdn-aj_redrel_c hdn-cl_dsh_c hdn-cl_prnth_c hdn-n_prnth_c hdn-np_app-idf-p_c hdn-np_app-idf_c hdn-np_app-nbr_c hdn-np_app-num_c hdn-np_app-pr_c hdn-np_app-r-pr_c hdn-np_app-r_c hdn-np_app_c hdn_bnp-num_c hdn_bnp-pn_c hdn_bnp-prd_c hdn_bnp-qnt_c hdn_bnp-sg-jmod_c hdn_bnp-sg-nmod_c hdn_bnp-sg-nomod_c hdn_bnp-vger_c hdn_bnp_c hdn_color_c hdn_np-num_c hdn_optcmp_c j-aj_frg_c j-j_crd-att-t_c j-j_crd-prd-im_c j-j_crd-prd-m_c j-j_crd-prd-t_c j-n_crd-t_c j_att_dlr j_enough-wc_dlr j_enough_dlr j_frg_c j_n-ed_c j_n-minut_dlr j_n-pre_odlr j_sbrd-pre_c jpr-jpr_crd-im_c jpr-jpr_crd-m_c jpr-jpr_crd-t_c jpr-vpr_crd-im_c jpr-vpr_crd-m_c jpr-vpr_crd-t_c mnp_deg-prd_c mnp_deg_c mrk-nh_atom_c mrk-nh_cl_c mrk-nh_evnt_c mrk-nh_n_c mrk-nh_nom_c n-hdn_cpd_c n-hdn_j-n-cpd_c n-j_crd-t_c n-j_j-cpd_c n-j_j-t-cpd_c n-n_crd-2-t_c n-n_crd-3-t_c n-n_crd-asym-t_c n-n_crd-asym2-t_c n-n_crd-div-t_c n-n_crd-im_c n-n_crd-m_c n-n_crd-nc-m_c n-n_crd-t_c n-n_num-seq_c n-num_mnp_c n-v_j-cpd_c n_bipart_dlr n_det-mnth_dlr n_det-wkdy_dlr n_mnp_c n_ms-cnt_ilr n_ms_ilr n_n-ed_odlr n_pl-cur_ilr n_pl_olr n_sg_ilr nb-aj_frg_c non_third_sg_fin_v_rbst np-aj_frg_c np-aj_j-frg_c np-aj_rorp-frg_c np-cl_indef_c np-cl_numitem_c np-hdn_cpd-pr_c np-hdn_cpd_c np-hdn_cty-cpd_c np-hdn_nme-cpd_c np-hdn_num-cpd_c np-hdn_ttl-cpd-pl_c np-hdn_ttl-cpd_c np-np_crd-i-t_c np-np_crd-i2-t_c np-np_crd-i3-t_c np-np_crd-im_c np-np_crd-m_c np-np_crd-nc-m_c np-np_crd-nc-t_c np-np_crd-t_c np-prdp_vpmod_c np_adv-mnp_c np_adv-yr_c np_adv_c np_cnj-frg_c np_frg_c np_nb-frg_c np_prt-poss_c np_voc-post_c np_voc-pre_c num-n_mnp_c num_det_c num_prt-det-nc_c num_prt-det-of_c num_prt-nc_c num_prt-of_c pp-aj_frg_c pp-pp_crd-im_c pp-pp_crd-m_c pp-pp_crd-t_c pp_frg_c ppr-ppr_crd-im_c ppr-ppr_crd-m_c ppr-ppr_crd-t_c r_cl-frg_c r_dsc-frg_c r_int-frg_c r_scp-frg_c root_bridge root_command root_conj root_decl root_formal root_frag root_inffrag root_informal root_lex root_non_idiom root_phr root_question root_robust root_robust_frag root_robust_s root_spoken root_spoken_frag root_standard root_strict root_subord sb-hd_mc_c sb-hd_nmc_c sb-hd_q_c sp-hd_hc-cmp_c sp-hd_hc_c sp-hd_n_c v-v_crd-fin-ncj_c v-v_crd-nfin-ncj_c v_3s-fin_olr v_aux-advadd_dlr v_aux-ell-ref_dlr v_aux-ell-xpl_dlr v_aux-sb-inv_dlr v_aux-tag_dlr v_cond-inv_dlr v_dat_dlr v_inv-quot_dlr v_j-nb-intr_dlr v_j-nb-pas-ptcl_dlr v_j-nb-pas-tr_dlr v_j-nb-prp-tr_dlr v_j-nme-intr_dlr v_j-nme-tr_dlr v_n3s-bse_ilr v_nger-intr_dlr v_nger-pp_dlr v_nger-tr_dlr v_np-prtcl_dlr v_pas-cp_odlr v_pas-dat_odlr v_pas-p-t_odlr v_pas-p_odlr v_pas-prt-t_odlr v_pas_odlr v_prp-nf_olr v_prp_olr v_psp_olr v_pst_olr v_v-co_dlr v_v-counter_dlr v_v-mis_dlr v_v-pre_dlr v_v-re_dlr v_v-un_dlr vp-vp_crd-fin-im_c vp-vp_crd-fin-m_c vp-vp_crd-fin-t_c vp-vp_crd-nfin-im_c vp-vp_crd-nfin-m_c vp-vp_crd-nfin-t_c vp_cp-sb-inf_c vp_cp-sb_c vp_fin-frg_c vp_nfin-frg_c vp_np-ger_c vp_rc-redrel_c vp_sbrd-prd-aj_c vp_sbrd-prd-ell_c vp_sbrd-prd-pas_c vp_sbrd-prd-prp_c vp_sbrd-pre-lx_c vp_sbrd-pre_c vppr-vppr_crd-im_c vppr-vppr_crd-m_c vppr-vppr_crd-t_c w-w_fw-seq-m_c w-w_fw-seq-t_c w_asterisk-pre_plr w_asterisk_plr w_bang_plr w_comma-nf_plr w_comma-rp_plr w_comma_plr w_double_semicol_plr w_dqleft_plr w_dqright_plr w_drop-ileft_plr w_drop-iright_plr w_hyphen_plr w_italics_dlr w_italleft_plr w_italright_plr w_lbrack_plr w_lparen_plr w_period_plr w_qmark-bang_plr w_qmark_plr w_qqmark_plr w_rbrack_plr w_rparen_plr w_semicol_plr w_sqleft_plr w_sqright_plr w_threedot_plr xp_brck-pr_c".split(" "))
half_erg_rules = set()
for r in erg_rules:
    half_erg_rules.add(r.split("_")[0] + "_" + r.split("_")[-1])

VOWELS = ['a', 'e', 'i', 'o', 'u']

with open("./erg_info/erg_rule_hdness.json", "r") as f:
    erg_rule2hdness = json.load(f)
erg_rule2hdness_tmp = erg_rule2hdness.copy()
for r in erg_rule2hdness_tmp:
    r_half = r.split("_")[0] + "_" + r.split("_")[-1]
    if r_half in erg_rule2hdness:
        if erg_rule2hdness[r_half] != 'unk':
            if erg_rule2hdness[r] != erg_rule2hdness[r_half]:
                if erg_rule2hdness[r][-3:] == 'nhd':
                    erg_rule2hdness[r_half] = erg_rule2hdness[r]
                else:
                    pass
#                     print (r, erg_rule2hdness[r], erg_rule2hdness[r.split("_")[0]])
        else:
            erg_rule2hdness[r_half] = erg_rule2hdness[r]
    else:
        erg_rule2hdness[r_half] = erg_rule2hdness[r]
# input()
    

prop_abbrev = {"sf":"sf","tense":"tn","mood":"md","perf":"pf","pers":"ps","num":"nm",
               "ind":"in","gend":"gd","pt":"pt","prog":"pg","instance":"ep","cvarsort":"cv"}
value_abbrev = {"COMM":"CM","PROP":"P","PROP-OR-QUES":"PQ","QUES":"Q",
                "FUT":"FU","PAST":"PA","PRES":"PR","TENSED":"TN","UNTENSED":"UT",
                "INDICATIVE":"I","SUBJUNCTIVE":"S",
                "+":"+","-":"-",
                "1":"1","2":"2","3":"3",
                "PL":"PL","SG":"SG",
                "F":"F","M":"M","M-OR-F":"MF","N":"N",
                "REFL":"RF","STD":"ST","ZERO":"ZR"
               }
# "md:I", "pf:-", "pg:-", "sf:P", "tn:PR", "arg1:SG&3", "arg2:-N&-P"
copula2trg = {"be_c_is": [["tn:PR"]],
              "be_inv_is": [["tn:PR"]],
              "be_c_be": [["tn:UT", "pf:-"], ["tn:FU", "pf:-"]],
              "be_c_was": [["tn:PA"]],
              "be_inv_was": [["tn:PA"]],
              "be_c_are": [["tn:PR"]],
              "be_inv_are": [["tn:PR"]],
              "be_c_am": [["tn:PR"]],
              "will_aux_pos":  [["tn:FU"]],
              "has_aux": [["tn:PR", "pf:+"]],
              "had_aux": [["tn:PA", "pf:+"], ["md:S", "pf:+"]],
              "be_c_been": [["pf:+"]],
              "be_c_were": [["tn:PA"], ["md:S", "pg:+"]],
              "be_inv_were": [["tn:PA"], ["md:S", "pg:+"]],
              "do1_pos": [["md:I", "tn:PR", "pf:-", "pg:-"]],
              "have_fin_aux": [["tn:PR", "pf:+", "md:I"]],
              "be_c_is_cx_3": [["tn:PR"]],
              "will_aux_pos_cx_3": [["tn:FU"]],
              "be_c_am_cx_3": [["tn:PR"]],
              "have_bse_aux": [["pf:+"]],
              "be_c_being": [["pg:+"]],
              "have_fin_aux_cx_3": [["tn:PR", "pf:+", "md:I"]],
              "be_c_are_cx_2": [["tn:PR"]],
              "did1_pos": [["md:I", "tn:PA", "pf:-", "pg:-"]],
              "be_inv_is": [["tn:PR", "md:I"]],
              "does1_pos": [["md:I", "tn:PR", "pf:-", "pg:-"]],
              "have_aux_prp": [["pf:+", "pg:+", "tn:UT"]],
              "had_aux_cx_3": [["tn:PA", "pf:+"], ["md:S", "pf:+"]],
              "to_c_prop": [["tn:UT"]]
              }

compl2trg = {"that_c": [["tn:PA"], ["tn:PR"], ["tn:FU"], ["tn:TN"]],
             "that_r": [["tn:PA"], ["tn:PR"], ["tn:FU"], ["tn:TN"]],
             "which_r": [["tn:PA"], ["tn:PR"], ["tn:FU"], ["tn:TN"]],
             "who2": [["tn:PA"], ["tn:PR"], ["tn:FU"], ["tn:TN"]],
             "whether_c_fin": [["tn:PA", "sf:Q"], ["tn:PR", "sf:Q"], ["tn:FU", "sf:Q"], ["tn:TN", "sf:Q"]]}

trgPred2semEmt = {"part_of": ["of"], "dofm": ["of"], "dofw": ["of"], "holiday": ["of"],
                  "mofy": ["of"], "place": ["of"],
                  "comp_too": ["to", "for"], "comp_enough": ["to", "for"],
                  "all+too":  ["to", "for"], "comp_not+so":  ["to", "for"],
                  "comp_so": ["that"], "relative_mod": ["that"],
                  "ellipsis_expl": ["there"],
                  "subord": ['which', 'that', 'who2'],
                  "comp": ["than"], "comp_less": ["than"],
                  "comp_equal": ["as"],
                  "generic_entity": ["of"]
                 }

byTrg2preTerm = {"parg_d": ["by_pass_p"],
                 "nominalization": ["of_prtcl"]}
byOf = ["by_pass_p", "of_prtcl"]


predSemEmts = set("of to for that that there which that who by than as".split(" "))


subj2objPron = {"we": "us", "i": "me", "you": "you", "she": "her", "he": "him", "they": 'them', "it": "it"}
obj2subjPron = {v: k for k, v in subj2objPron.items()}

# semEmt2trgPred = {"who"}

# semEmt2trgProps = {"be_c_is": [["tn:PR"]],
#                   "be_c_be": [["tn:UT", "pf:-"], ["tn:FU", "pf:-"]],
#                   "be_c_was": [["tn:PA"]],
#                   "be_c_are": [["tn:PR"]],
#                   "be_c_am": [["tn:PR"]],
#                   "will_aux_pos":  [["tn:FU"]],
#                   "has_aux": [["tn:PR", "pf:+"]],
#                   "had_aux": [["tn:PA", "pf:+"], ["md:S", "pf:+"]],
#                   "be_c_been": [["pf:+"]],
#                   "be_c_were": [["tn:PA"], ["md:S", "pg:+"]],
#                   "do1_pos": [["md:I", "tn:PR", "pf:-", "pg:-"]],
#                   "have_fin_aux": [["tn:PR", "pf:+", "md:I"]],
#                   "be_c_is_cx_3": [["tn:PR"]],
#                   "will_aux_pos_cx_3": [["tn:FU"]],
#                   "be_c_am_cx_3": [["tn:PR"]],
#                   "have_bse_aux": [["pf:+"]],
#                   "be_c_being": [["pg:+"]],
#                   "have_fin_aux_cx_3": [["tn:PR", "pf:+", "md:I"]],
#                   "be_c_are_cx_2": [["tn:PR"]],
#                   "did1_pos": [["md:I", "tn:PA", "pf:-", "pg:-"]],
#                   "be_inv_is": [["tn:PR", "md:I"]],
#                   "does1_pos": [["md:I", "tn:PR", "pf:-", "pg:-"]],
#                   "have_aux_prp": [["pf:+", "pg:+", "tn:UT"]],
#                   "had_aux_cx_3":  [["tn:PA", "pf:+"], ["md:S", "pf:+"]],
                   
#                   "to_c_prop": [["tn:UT"]],
                   
#                   }


def load_config(config_name, redwdOrGw):
#     configs_name = ["dense", "standard"]
    config_filepath = "./config/config_{}.json".format(redwdOrGw+"-"+config_name)
    with open(config_filepath, "r", encoding='utf-8') as f:
        config = json.load(f)
    return config, config_filepath


def write_figs_err(dmrs_nxDG, deriv_nxDG, snt, name = "err"):
    print ("err-ed snt:", snt)
    if dmrs_nxDG:
        write_dmrsFig_err(dmrs_nxDG.copy(), name)
    if deriv_nxDG:
        write_derivFig_err(deriv_nxDG.copy(), name)
    
def write_dmrsFig_err(dmrs_nxDG, name = "err"):
    erg_digraphs = dg_util.Erg_DiGraphs()
    erg_digraphs.init_dmrs_from_nxDG(dmrs_nxDG, draw = True)
#     print (erg_digraphs.dmrs_dg.out_edges(data = 'label'))
#     print (erg_digraphs.dmrs_dg.out_edges())
#     print (erg_digraphs.dmrs_dg.out_edges)
    save_path = "./figures/dmrs_{}_".format(name)+ time.asctime( time.localtime(time.time()) ).replace(" ", "-") +".png"
    ag = to_agraph(erg_digraphs.dmrs_dg)
    ag.layout('dot')
    ag.draw(save_path)
    print ("err dmrs drawn:", save_path)
    
def write_derivFig_err(deriv_nxDG, name = "err"):
    erg_digraphs = dg_util.Erg_DiGraphs()
    erg_digraphs.init_erg_deriv_from_nxDG(deriv_nxDG, draw = True)
    save_path = "./figures/deriv_{}_".format(name) + time.asctime( time.localtime(time.time()) ).replace(" ", "-") + ".png" 
    ag = to_agraph(erg_digraphs.deriv_dg)
    ag.layout('dot')
    ag.draw(save_path)
    print ("err deriv drawn:", save_path)

    

def is_syn_construction(rule_name):
    return rule_name.split("&")[0].split("^")[0].endswith("_c") and (rule_name in erg_rules\
        or rule_name in half_erg_rules)

def get_deriv_leftRight_dgtrs(deriv_nxDG, node):
    unary_daughter = [targ for src, targ, lbl in deriv_nxDG.out_edges(
        nbunch = [node],
        data = 'label')
        if lbl == 'U']
    if unary_daughter:
        return (unary_daughter[0],)
    left_daughter = [targ for src, targ, lbl in deriv_nxDG.out_edges(
        nbunch = [node],
        data = 'label')
        if lbl == 'L']
    right_daughter = [targ for src, targ, lbl in deriv_nxDG.out_edges(
        nbunch = [node],
        data = 'label')
        if lbl == 'R']
    if left_daughter and right_daughter:
        return (left_daughter[0], right_daughter[0])
    else:
        return ()
    
def get_lemma_pos(node_prop):
    # extract lemma pos
    if node_prop['instance'][0] != "_":
        return node_prop['instance'], "S"
    elif "unknown" in node_prop['instance'] or node_prop['instance'] in ["_nowhere_near_x_deg",
                                                                         "_pay_per_view_a_1"]:
        pred_lemma, pred_pos, *_ = node_prop['instance'].rsplit("_", 2)
        pred_lemma = pred_lemma.replace('+',' ')[1:]
        # print (pred_lemma, pred_pos)
    else:
        _, pred_lemma, pred_pos, *_ = node_prop['instance'].split("_")
        pred_lemma = pred_lemma.replace('+',' ')
        if pred_pos == 'dir': pred_pos = 'p'
        if pred_pos == 'state': pred_pos = 'p'
    if pred_pos not in 'a v n q x p c':
        print (node_prop['instance'])
        print (pred_lemma, pred_pos)
        pred_lemma, pred_pos, *_ = node_prop['instance'].rsplit("_", 2)
        pred_lemma = pred_lemma.replace('+',' ')[1:]
        print (pred_lemma, pred_pos)
        
    return pred_lemma, pred_pos

def get_pred_prtcl(pred):
    if pred[0] == "_":
        pred_split = pred.split("_")
        if len(pred_split) > 3:
            _, pred_lemma, pred_pos, *_, pred_sense = pred_split
            if not pred_sense.isnumeric():
                # separate "-"?
                if pred_sense in particles:
                    return pred_sense
    return None

def get_surface_ofNode(node_prop,sentence):
    # extract surface
    anchor_from, anchor_to = anchorstr2anchors(node_prop['lnk'])
    surface = sentence[anchor_from:anchor_to].lower()
    return surface

def get_surface_ofNodes(node_props, sentence):
    min_anc, max_anc = (2147483647,0)
    # extract surface
    for node_prop in node_props:
        anchor_from, anchor_to = anchorstr2anchors(node_prop['lnk'])
        min_anc = min(min_anc, anchor_from)
        max_anc = max(max_anc, anchor_to)
    surface = sentence[min_anc:max_anc]
    return surface

def anchorstr2anchors(anchor_str):
    anchor_from, anchor_to = anchor_str.split(":")
    anchor_from = int(anchor_from[1:])
    anchor_to = int(anchor_to[:-1])
    return (anchor_from,anchor_to)

def anchors2anchorstr(anchors):
    return "<" + str(anchors[0]) + ":" + str(anchors[1]) + ">"

def sort_anchors(anchors2nodeRepIdxEdge):
    def cmp_anchors(a, b):
        if a[0] < b[0] and a[1] < b[1]:
            return -1
        elif a[0] > b[0] and a[1] > b[1]:
            return 1
        else:
            return 0
    sorted_nodeRepIdxEdge = tuple(v for k, v in sorted(anchors2nodeRepIdxEdge.items(),
                                                       key = cmp_to_key(lambda x,y:
                                                                        cmp_anchors(x[0],y[0]))))
    return sorted_nodeRepIdxEdge

def get_uncovered_anchors(anchors, snt_len):
    # return list of intervals which anchors does not cover
    uncovered_anchors = []
    an = [(1,3),(6,9),(8,11),(15,20),(10,12),(11,13)]
    interval_mark = [0 for _ in range(0,snt_len+1)]
    intervals = [0 for _ in range(0,snt_len+1)]

    for an1,an2 in anchors:
        try:
            interval_mark[an1] += 1
            interval_mark[an2] -= 1
        except:
            print (an1,an2,snt_len)

    prev = 0
    for i in range(len(interval_mark)):
        intervals[i] = prev + interval_mark[i]
        prev = intervals[i]

    start,end = (None,None)
    for i in range(len(intervals)):
        if intervals[i] == 0:
            if start == None:
                start = i
            elif i == snt_len:
                end = i
                uncovered_anchors.append((start,end))
        else:
            if start != None:
                end = i
                uncovered_anchors.append((start,end))
                start = None
    return uncovered_anchors

def node_toString(node,node_prop,dmrs_nxDG,underspecLemma = False,underspecCarg=True, add_posSpecInfo = True, lexicalized = False, forSurfGen = False):
    prop_abbrev = {"sf":"sf","tense":"tn","mood":"md","perf":"pf","pers":"ps","num":"nm",
                   "ind":"in","gend":"gd","pt":"pt","prog":"pg","instance":"ep","cvarsort":"cv","carg":"cg"}
    new_prop_abbrev = {**prop_abbrev, **{"lemma":"lm","pos":"pos","verb_form":"vf","instance":"ep","num":"num","arg1_cvarsort":"a1cv","comp_superl_abs":"cOrs", "parg_d":"psv"}}
    value_abbrev = {"COMM":"CM","PROP":"P","PROP-OR-QUES":"PQ","QUES":"Q",
                    "FUT":"FU","PAST":"PA","PRES":"PR","TENSED":"TN","UNTENSED":"UT",
                    "INDICATIVE":"I","SUBJUNCTIVE":"S",
                    "+":"+","-":"-",
                    "1":"1","2":"2","3":"3",
                    "PL":"PL","SG":"SG",
                    "F":"F","M":"M","M-OR-F":"MF","N":"N",
                    "REFL":"RF","STD":"ST","ZERO":"ZR",
                    }
    new_value_abbrev = {**value_abbrev, **{"comp":"C","superl":"S","abs":"ABS","active":"ACT","passive":"PSV",
                                           "past_part":"PSP","pres_part":"PRP","base_form/to_el":"BI/TO_EL","bare_past":"BP","third_per_sg":"3PS",
                                           "_are/do_el":"_ARE/DO_EL","_am/do_el":"_AM/DO_EL","_were/did_el":"_WERE/DID_EL","_was/did_el":"_WAS/DID_EL",
                                           "v":"v",
                                           "SG":"SG","PL":"PL","abs":"ABS",
                                           "jj":"jj","jr":"jr","jjs": "jjs","cd": "cd","nn": "nn","nns": "nns","nnps": "nnps","nnp": "nnp","fw": "fw",
                                           "rb": "rb","vb": "vb","vbg": "vbg","vbn": "vbn","vbp": "vbp","vbz": "vbz"}}
#     if node_prop['instance'][0] == '_':
#         pred_lemma,pred_pos = get_lemma_pos(node_prop)
#         assigned_node_key = assign_lexical_node_key(node,node_prop,pred_lemma,pred_pos,dmrs_nxDG,underspecLemma)
#         return _node_key_toString(assigned_node_key,prop_abbrev,value_abbrev)
#     # handle 'be's/'do's
#     elif node_prop['instance'] in ["ellipsis_expl","ellipsis_ref"]:
#         assigned_node_key = assign_lexical_node_key(node,node_prop,node_prop['instance'],'v',dmrs_nxDG,underspecLemma)
#         return _node_key_toString(assigned_node_key,prop_abbrev,value_abbrev)
#     else:
    return _node_prop_toString(dmrs_nxDG, node, node_prop,new_prop_abbrev,new_value_abbrev,
                               underspecLemma, underspecCarg, add_posSpecInfo = add_posSpecInfo,
                              lexicalized = lexicalized, forSurfGen = forSurfGen)
    
def _node_key_toString(node_key,prop_abbrev,value_abbrev):   
    new_prop_abbrev = {**prop_abbrev, **{"lemma":"lm","pos":"pos","verb_form":"vf","instance":"ep","num":"num","arg1_cvarsort":"a1cv","comp_superl_abs":"cOrs"}}
    new_value_abbrev = {**value_abbrev, **{"comp":"C","superl":"S","abs":"ABS",
                                           "past_part":"PSP","pres_part":"PRP","base_form/to_el":"BI/TO_EL","bare_past":"BP","third_per_sg":"3PS",
                                           "_are/do_el":"_ARE/DO_EL","_am/do_el":"_AM/DO_EL","_were/did_el":"_WERE/DID_EL","_was/did_el":"_WAS/DID_EL",
                                           "v":"v",
                                           "SG":"SG","PL":"PL","abs":"ABS",
                                           "jj":"jj","jr":"jr","jjs": "jjs","cd": "cd","nn": "nn","nns": "nns","nnps": "nnps","nnp": "nnp","fw": "fw",
                                           "rb": "rb","vb": "vb","vbg": "vbg","vbn": "vbn","vbp": "vbp","vbz": "vbz"}}
    try:
        node_prop_list = [new_prop_abbrev[prop]+":"+new_value_abbrev[str(value)]
                          for prop,value in node_key
                          if not prop in {'instance','ARG1-*','lemma','lnk','carg','cvarsort'}]
    
    except:
        print(node_key)
        print (new_prop_abbrev)
        
    node_prop_list += [new_prop_abbrev[prop]+":"+value for prop,value in node_key if prop in {'instance','ARG1-*','lemma','cvarsort'}]
    sorted_node_prop_list = sorted(node_prop_list)
    return (";").join(sorted_node_prop_list)

def _node_prop_toString(dmrs_nxDG, node, node_prop, prop_abbrev, value_abbrev, underspecLemma = False, underspecCarg = True, add_posSpecInfo = True, lexicalized = False, forSurfGen = False):
    # print (node_prop)
    exc_prop = ['lnk','instance','cvarsort', 'carg']
    extra_prop = ["ordered_ext_nodes", "extracted", "derivRule", "replaced_nodes_repl", "order_char", "order_int"]
    lex_prop = ['sf']
    pred = node_prop['instance']
    if underspecLemma:
        if pred[0] == "_" and (pred.split("_")[2] in ['n','a','v'] or unknown2pos.get(pred.split("_")[2]) in ['n','a','v']):
            if len(pred.split("_")) >= 4 and pred.split("_")[3] == "modal":
                pred = "_[usp]_"+unknown2pos.get(pred.split("_")[2])+"_modal"
#                 print ("modal")
            else:
                pred = "_[usp]_"+unknown2pos.get(pred.split("_")[2])
            
    node_prop_list = ["ep:"+pred]
    if lexicalized:
#         node_prop_list += [prop_abbrev[prop]+":"+value_abbrev[str(node_prop[prop])]
#                             for prop in node_prop if prop in lex_prop]
        sorted_node_prop_list = sorted(node_prop_list)
        return (";").join(sorted_node_prop_list)
#         if pred[0] == '_':
#             pass
    if not underspecCarg and 'carg' in node_prop: 
        node_prop_list += ['cg:'+node_prop['carg']]
    elif underspecCarg and 'carg' in node_prop:
        node_prop_list += ['cg:'+"[usp]"]
    node_prop_list += [prop_abbrev[prop]+":"+str(node_prop[prop])
                        for prop in node_prop if prop in {'cvarsort'}]
    node_prop_list += [prop_abbrev[prop]+":"+value_abbrev[str(node_prop[prop])]
                            for prop in node_prop if not prop in exc_prop + extra_prop]
    if forSurfGen:
        add_posSpecInfo = True
    if add_posSpecInfo:
        # adj/adv
        if pred[0] == '_':
            node_prop_list += [f'pos:{unknown2pos.get(pred.split("_")[2])}']
            if pred.split("_")[2] == 'a' or unknown2pos.get(pred.split("_")[2]) == 'a':
                # consider cvcarsort of arg1 of current node to determine adj vs adv
                out_edges = dmrs_nxDG.out_edges(nbunch=[node],data=True)
                adjadv_prop_dict = {"arg1_cvarsort": 'abs', # ~1800 cases
                                    "comp_superl_abs": 'abs'}
                for out_edge in out_edges:
                    edge_label = out_edge[2]['label']
                    if "ARG1/EQ" in edge_label:
                        cvarsort = dmrs_nxDG.nodes[out_edge[1]].get('cvarsort')
                        if cvarsort:
                            adjadv_prop_dict["arg1_cvarsort"] = cvarsort
                        else:
                            adjadv_prop_dict["arg1_cvarsort"] = 'x'
                            print ("no cvarsort:", dmrs_nxDG.nodes[out_edge[1]])
                        break
                # consider if there is any incoming comp/su of arg1 of current node
                in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
                for in_edge in in_edges:
                    edge_label = in_edge[2]['label']
                    edge_source_pred = dmrs_nxDG.nodes[in_edge[0]]['instance']
                    if edge_source_pred in ['comp','superl'] and "ARG1" in edge_label:
                        adjadv_prop_dict["comp_superl_abs"] = edge_source_pred
                        break
                node_prop_list += [prop_abbrev[prop]+":"+value_abbrev[v]
                                    if prop != "arg1_cvarsort"
                                    else prop_abbrev[prop]+":"+v
                                   for prop, v in adjadv_prop_dict.items()]
            elif pred.split("_")[2] == 'v' or unknown2pos.get(pred.split("_")[2]) == 'v':
                # consider passive voice
                in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
                verb_prop_dict = {"parg_d": 'active'}
                for in_edge in in_edges:
                    edge_label = in_edge[2]['label']
#                     try:
                    edge_source_pred = dmrs_nxDG.nodes[in_edge[0]]['instance']
#                     except:
#                         print (dmrs_nxDG.nodes[in_edge[0]])
#                         input()
                    if "ARG1/EQ" in edge_label and edge_source_pred == 'parg_d':
                        verb_prop_dict["parg_d"] = "passive"
                        break
                node_prop_list += [prop_abbrev[prop]+":"+value_abbrev[v]
                                   for prop, v in verb_prop_dict.items()]    
    if forSurfGen:
        if pred[0] == '_':
            if pred.split("_")[2] == 'a' or unknown2pos.get(pred.split("_")[2]) == 'a':
                pass
            elif pred.split("_")[2] == 'v' or unknown2pos.get(pred.split("_")[2]) == 'v':
                # consider 3sg for tn:PR
                sg3 = "-"
                if "tn:PR" in node_prop_list and "pg:-" in node_prop_list and "pf:-" in node_prop_list\
                    and "sf:P" in node_prop_list:
                    for src, targ, lbl in dmrs_nxDG.out_edges(node, data = 'label'):
                        if "ARG1" in lbl and "psv:ACT" in node_prop_list:
                            if str(dmrs_nxDG.nodes[targ].get('pers')) == '3' and dmrs_nxDG.nodes[targ].get('num') == 'SG':
                                sg3 = "+"
                node_prop_list += ["3sg:" + sg3]
                # consider negation
                neg = "+"
                in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
                for in_edge in in_edges:
                    edge_label = in_edge[2]['label']
                    edge_source_pred = dmrs_nxDG.nodes[in_edge[0]]['instance']
                    if edge_label == "ARG1/H" and edge_source_pred == 'neg':
                        neg = "-"
                        break
                node_prop_list += ["neg:" + neg]
        
    sorted_node_prop_list = sorted(node_prop_list)
    return (";").join(sorted_node_prop_list)

def copula_trg_node_prop_toString(node_prop):
    # print (node_prop)
    relevant_props1 = ["mood", "perf"]
    relevant_props2 = ["prog", "sf", "tense"]
    sorted_node_prop_list = [prop_abbrev[prop]+":"+value_abbrev[str(node_prop[prop])]
                             for prop in relevant_props1]
    if node_prop.get('instance'):
        if len(node_prop.get('instance').split("_")) >= 3:
            if node_prop.get('instance').split("_")[2] in unknown2pos:
                sorted_node_prop_list += [f'pos:{unknown2pos[node_prop.get("instance").split("_")[2]]}']
    sorted_node_prop_list += [prop_abbrev[prop]+":"+value_abbrev[str(node_prop[prop])]
                             for prop in relevant_props2]
    return (";").join(sorted_node_prop_list)

def get_semicanon_fromSubgr(subgraph, underspecLemma = False):
    subgraph_preds = [subgraph.nodes[node]['instance'] for node in subgraph.nodes()]
    return get_semicanonical_form(subgraph_preds, underspecLemma)

def get_semicanonical_form(subgraph_preds, underspecLemma = False, from_exact_semi = False):
    # receive list of predicates
    if from_exact_semi: subgraph_preds_cnter = subgraph_preds
    else: subgraph_preds_cnter = Counter(subgraph_preds).items()
    if underspecLemma:
        usp_subgraph_preds_cnter = Counter()
        for pred,cnt in subgraph_preds_cnter:
            if pred[0] == "_":
                try:
                    if pred.split("_")[2] in ['n','a','v']:
                        usp_subgraph_preds_cnter["_[usp]_"+pred.split("_")[2]] += cnt
                    elif unknown2pos.get(pred.split("_")[2]) in ['n','a','v']: 
                        usp_subgraph_preds_cnter["_[usp]_"+unknown2pos.get(pred.split("_")[2])] += cnt
                    else:
                        usp_subgraph_preds_cnter[pred] += cnt
                except:
                    usp_subgraph_preds_cnter[pred] += cnt
            else:
                usp_subgraph_preds_cnter[pred] += cnt
        return str(tuple(sorted(usp_subgraph_preds_cnter.items(), key = lambda x: x[0])))
    
    else:
#         print (subgraph_preds_cnter)
        return str(tuple(sorted(subgraph_preds_cnter, key = lambda x: x[0])))
#         return frozenset(subgraph_preds_cnter.items())

def get_canonical_form(node_ind_subgraph,dmrs_nxDG,dummy_nodes=[],sentence = None,extract_surfacemap = False, underspecLemma = False, underspecCarg = False, lexicalized = False, forSurfGen = True):
    def get_nodeRep(target_node,curr_hop,nhop,nodeRep):
        # bfs
        node2visited = {node: False for node in node_ind_subgraph.nodes()}
        node_queue = [(target_node,0)]
        prev_hop = 0
        while len(node_queue) > 0:
            curr_node,curr_hop = node_queue.pop(0)
            curr_hop += 1
            if curr_hop <= nhop:
#                 sorted_node_edgeReps = get_sorted_node_edgeReps(curr_node,node2visited)
                sorted_node_edgeReps = node2sorted_node_edgeReps[curr_node]
                if (curr_hop > prev_hop):
                    nodeRep += ">>"
                if sorted_node_edgeReps != []:
                    if (curr_hop == prev_hop):
                        nodeRep += "*"
                    sorted_nodes, sorted_edgeReps = zip(*sorted_node_edgeReps)
                    node_queue += list(zip(sorted_nodes,[curr_hop]*len(sorted_node_edgeReps)))
                    edgesRep = ("&").join(sorted_edgeReps)
                    nodeRep += edgesRep
            node2visited[curr_node] = True
            prev_hop = curr_hop
        return nodeRep
    
    def get_sorted_node_edgeReps(curr_node,node2visited=None):
        node_out_edgeReps = [(targ,node2string[targ]+"$"+lbl)
                             for src,targ,lbl in node_ind_subgraph.out_edges(curr_node,data='label')
                             # if not node2visited[targ]
                            ]
        node_in_edgeReps = [(src,node2string[src]+"$-"+lbl)
                            for src,targ,lbl in node_ind_subgraph.in_edges(curr_node,data='label')
                            # if not node2visited[src]
                           ]
        node_edgeReps = node_out_edgeReps + node_in_edgeReps
        sorted_node_edgeReps = sorted(node_edgeReps, key = lambda t:t[1])
        return sorted_node_edgeReps
    add_posSpecInfo = True
    if lexicalized:
        add_posSpecInfo = False
    node2string = {node: node_toString(node,node_prop,dmrs_nxDG,underspecLemma,underspecCarg,
                                       add_posSpecInfo = add_posSpecInfo,
                                       lexicalized = lexicalized, forSurfGen = forSurfGen)
                   if not node in dummy_nodes else "dummy"
                   for node, node_prop in node_ind_subgraph.nodes(data=True)}
#     print (node2string)
    node2sorted_node_edgeReps = {node: get_sorted_node_edgeReps(node)
                                 for node in node_ind_subgraph.nodes()}
    nhop = 2
    nodeReps = tuple(get_nodeRep(node,
                                 curr_hop=1,
                                 nhop=nhop,
                                 nodeRep=node2string[node])
                     for node,node_prop in node_ind_subgraph.nodes(data=True))
    subgraph_nodes = tuple(node for node in node_ind_subgraph.nodes())
    if extract_surfacemap:
        surfaceMaps = tuple(sentence[anchorstr2anchors(node_ind_subgraph.nodes[node]['lnk'])[0]:
                                     anchorstr2anchors(node_ind_subgraph.nodes[node]['lnk'])[1]]
                            for node in node_ind_subgraph.nodes())
    else:
        surfaceMaps = tuple(None for node in node_ind_subgraph.nodes())
        
    zipped_nodeReps_nodes_surfaceMaps = zip(nodeReps,subgraph_nodes,surfaceMaps)
    sorted_nodeReps_nodes_surfaceMaps = sorted(zipped_nodeReps_nodes_surfaceMaps, key = lambda t:t[0])
    sorted_nodeReps,sorted_nodes,sorted_surfaceMaps = zip(*sorted_nodeReps_nodes_surfaceMaps)
    canonical_form = tuple(tuple(nodeRep.split(">>")) for nodeRep in sorted_nodeReps)
    return canonical_form,sorted_nodes,sorted_surfaceMaps

def get_equalanchor_nodes(dmrs_nxDG, lexical_only = False):
    anchors2nodes = defaultdict(list)
    anchors2preds = defaultdict(list)
    # get anchors that correspond to lexical node or grammar predicate that would generate surface (e.g. pron)
    if lexical_only:
        lexical_anchors = set()
        for node, node_prop in dmrs_nxDG.nodes(data=True):
            if node_prop['instance'][0] == '_' or node_prop['instance'] in lexeme_pred or\
               node_prop['instance'] == 'pron' and (node_prop['pt'] != 'ZERO' or node_prop['pers'] == 1):
                anchor_from, anchor_to = anchorstr2anchors(node_prop['lnk'])
                lexical_anchors.add((anchor_from,anchor_to))
        # obtain nodes with same anchors as those in the extracted anchors
        for node, node_prop in dmrs_nxDG.nodes(data=True):
            anchor_from, anchor_to = anchorstr2anchors(node_prop['lnk'])
            if (anchor_from,anchor_to) in lexical_anchors and\
               (not node_prop['instance'] in structural_pred or\
                node_prop['instance'] == 'pron' and (node_prop['pt'] != 'ZERO' or node_prop['pers'] == 1)):
                # not node_prop['instance'] in lexical_mod_pred and\
                anchors2preds[(anchor_from,anchor_to)].append(node_prop['instance'])
                anchors2nodes[(anchor_from,anchor_to)].append(node)
    else:
        for node, node_prop in dmrs_nxDG.nodes(data=True):
            anchor_from, anchor_to = anchorstr2anchors(node_prop['lnk'])
            anchors2preds[(anchor_from,anchor_to)].append(node_prop['instance'])
            anchors2nodes[(anchor_from,anchor_to)].append(node)
    return anchors2nodes, anchors2preds

def r(a):
    return a

def get_node_neighbours(g, i, outOrIn = 'all'):
    out_neighbours, in_neighbours = (), ()
    if outOrIn in ['all', 'out']:
        out_edges = g.out_edges(i)
        if out_edges:
            out_neighbours = list(zip(*out_edges))[1]
#     print (i)
#     print ("o:", out_neighbours)
    if outOrIn in ['all', 'in']:
        in_edges = g.in_edges(i)
        if in_edges:
            in_neighbours = list(zip(*in_edges))[0]
#     print (in_neighbours, out_neighbours)
    neighbours = set(out_neighbours + in_neighbours)
#     print ("i:", in_neighbours)
    return neighbours

def gen_all_maxsizek_subgraph(dmrs_nxDG,k,sentence=None,subgraphs_semicanon=None,subgraphs_semicanon_usp=None,extract_surfacemap=False,relevant_only=True, get_uspCarg = False, lexicalized = False):
    
    def get_node_neighbours(node,dmrs_nxDG):
        out_edges = dmrs_nxDG.out_edges(node)
        in_edges = dmrs_nxDG.in_edges(node)
        neighbours = []
        if out_edges:
            neighbours += list(zip(*out_edges))[1]
        if in_edges:
            neighbours += list(zip(*in_edges))[0]
        return set(neighbours)
    
    def extend_subgraph(nodes_ofSubgraph,nodes_ofExtension,curr_node):
#         if (sentence == "I had made an order last week and then I looked at your site and saw that the same product was advertised at a cheaper price than what I had bought it for."):
#             print (nodes_ofSubgraph, nodes_ofExtension, curr_node)
        curr_subgraph_size = len(nodes_ofSubgraph)
        if (curr_subgraph_size <= k and curr_subgraph_size > 1):
            # print (len(nodes_ofSubgraph))
#             print (dmrs_nxDG.subgraph(nodes_ofSubgraph).nodes(data='instance'))
            # fast (rough) check of subgraph membership in document rules
            canonical_form = None
            surface_canonical_form = None
            usp_canonical_form = None
            uspCarg_canonical_form = None
            subgraph_preds = [dmrs_nxDG.nodes[node]['instance'] for node in nodes_ofSubgraph]
            if get_semicanonical_form(subgraph_preds) in subgraphs_semicanon:
                surface_canonical_form, sorted_nodes, surface_map\
                    = get_canonical_form(dmrs_nxDG.subgraph(nodes_ofSubgraph),
                                         dmrs_nxDG, sentence = sentence,
                                         extract_surfacemap = extract_surfacemap) 
                canonical_form, sorted_nodes, surface_map\
                    = get_canonical_form(dmrs_nxDG.subgraph(nodes_ofSubgraph),
                                         dmrs_nxDG, sentence = sentence,
                                         extract_surfacemap = extract_surfacemap, 
                                         lexicalized = lexicalized)
                all_sizei_nodesSubgraph[curr_subgraph_size].add((sorted_nodes,
                                                         (canonical_form, surface_canonical_form),
                                                                     surface_map))
                if get_uspCarg:
                    uspCarg_canonical_form, sorted_nodes_uspCarg, surface_map\
                        = get_canonical_form(dmrs_nxDG.subgraph(nodes_ofSubgraph),
                                         dmrs_nxDG, sentence = sentence,
                                         extract_surfacemap = extract_surfacemap, 
                                             underspecCarg = True, lexicalized = lexicalized) 
                    if canonical_form and not uspCarg_canonical_form:
                        # print (subgraphs_semicanon_usp)
                        print ("canonical_form and not uspCarg_canonical_form")
                        print (canonical_form, 4)
                        print (uspCarg_canonical_form, 5)
                        # input()
                    elif canonical_form and uspCarg_canonical_form:
                        canon2canon_uspCarg[canonical_form] = uspCarg_canonical_form
                    all_sizei_nodesSubgraph_uspCarg[curr_subgraph_size].add((sorted_nodes_uspCarg,
                                                                             (uspCarg_canonical_form, canonical_form),
                                                                             surface_map))
#             if "elliptical_n" in subgraph_preds:
#                 print (subgraph_preds)
#                 print (get_semicanonical_form(subgraph_preds, underspecLemma = True))
            if get_semicanonical_form(subgraph_preds, underspecLemma = True) in subgraphs_semicanon_usp:
                usp_canonical_form, sorted_nodes_usp, surface_map_usp\
                    = get_canonical_form(dmrs_nxDG.subgraph(nodes_ofSubgraph),
                                         dmrs_nxDG, sentence = sentence,
                                         extract_surfacemap = extract_surfacemap,
                                         underspecLemma = True, underspecCarg = True,
                                         lexicalized = lexicalized)
                if not surface_canonical_form:
                    surface_canonical_form, sorted_nodes, surface_map\
                        = get_canonical_form(dmrs_nxDG.subgraph(nodes_ofSubgraph),
                                             dmrs_nxDG, sentence = sentence,
                                             extract_surfacemap = extract_surfacemap) 
                all_sizei_nodesSubgraph_usp[curr_subgraph_size].add((sorted_nodes_usp,
                                                                     (usp_canonical_form, surface_canonical_form),
                                                                     surface_map_usp))
            if canonical_form and usp_canonical_form:
                canon2canon_usp[canonical_form] = usp_canonical_form
            if canonical_form and not usp_canonical_form:
                    # print (subgraphs_semicanon_usp)
                print (canonical_form, usp_canonical_form)
                print ("no semi canon")
                # input()
#             if get_uspCarg:
#                 if canonical_form and uspCarg_canonical_form:
#                     canon2canon_uspCarg[canonical_form] = uspCarg_canonical_form
#                 if canonical_form and not uspCarg_canonical_form:
#                     # print (subgraphs_semicanon_usp)
#                     print (canonical_form, uspCarg_canonical_form)
#                     input()
        if curr_subgraph_size < k:
            while nodes_ofExtension:
                chosen_node = nodes_ofExtension.pop()
                chosen_node_neighbours = {neighbour for neighbour in node2neighbours[chosen_node]
                                          if neighbour > curr_node}
                subgraph_neighbours = {neighbour for subgraph_node in nodes_ofSubgraph
                                             for neighbour in node2neighbours[subgraph_node]
                                             }
                chosen_node_excl_neighbours = chosen_node_neighbours - subgraph_neighbours
                nodes_ofExtension_new = nodes_ofExtension | chosen_node_excl_neighbours
                extend_subgraph(nodes_ofSubgraph | set([chosen_node]), nodes_ofExtension_new, curr_node)
    canon2canon_usp = defaultdict()
    canon2canon_uspCarg = defaultdict()
    all_sizei_nodesSubgraph = [set() for _ in range(k+1)]
    all_sizei_nodesSubgraph_usp = [set() for _ in range(k+1)]
    all_sizei_nodesSubgraph_uspCarg = [set() for _ in range(k+1)]
    node2neighbours = {node: get_node_neighbours(node,dmrs_nxDG) for node in dmrs_nxDG.nodes()}
    for node,node_prop in dmrs_nxDG.nodes(data=True):
        #neighbours_set = set(neighbours)
        nodes_ofExtension = {neighbour for neighbour in node2neighbours[node] if neighbour > node}
        extend_subgraph(set([node]),nodes_ofExtension,node)
    
    # assert len(all_sizei_subgraph) == len(set(all_sizei_subgraph))
    return all_sizei_nodesSubgraph, all_sizei_nodesSubgraph_usp, all_sizei_nodesSubgraph_uspCarg, canon2canon_usp, canon2canon_uspCarg
    
# def subgraph2node(subgraph_dmrs_nxDG, sorted_nodes, subgraph_nodes, subgraph_canon, replacing_node):
#     # sorted_nodes is essentially subgraph_nodes sorted according to nodeRep in canon
#     # Note: subgraph_nodes and node2nodeRepIdx should have the same keys
#     node2nodeRepIdx = {node: idx for idx,node in enumerate(sorted_nodes)} 
    
#     new_subgraph_dmrs_nxDG = subgraph_dmrs_nxDG # reference only
#     # not available during test
#     subgraph_nodes_anchors = [new_subgraph_dmrs_nxDG.nodes[node]['lnk'] for node in subgraph_nodes]
#     subgraph_nodes_anchors_set = set(subgraph_nodes_anchors)
#     # case of (weakly) disconnected
#     if len(subgraph_nodes_anchors_set) != 1:
#         # print (subgraph_nodes_anchors)
#         return False
    
#     new_subgraph_dmrs_nxDG.add_node(replacing_node, replaced_nodes = sorted_nodes, subgraph_canon = subgraph_canon,lnk=subgraph_nodes_anchors[0])
#     # identify the idx(position in canon) of source node (in a subgraph, which identifies each node in the suggraph) of each edge and add the idx to node2nodeRepIdx
# #     try:
#     new_outedges = [(replacing_node,targ,{'label':str(node2nodeRepIdx[src])+"-"+lbl})
#                     for node in subgraph_nodes
#                     for src,targ,lbl in new_subgraph_dmrs_nxDG.out_edges(node, data='label')
#                     if not targ in subgraph_nodes]
# #     except Exception as e:
# #         print (e)
# #         print (node2nodeRepIdx)
# #         print (subgraph_dmrs_nxDG.nodes)
#     new_inedges = [(src,replacing_node,{'label':lbl})
#                    for node in subgraph_nodes
#                    for src,targ,lbl in new_subgraph_dmrs_nxDG.in_edges(node, data='label')
#                    if not src in subgraph_nodes]
#     new_subgraph_dmrs_nxDG.add_edges_from(new_outedges)
#     new_subgraph_dmrs_nxDG.add_edges_from(new_inedges)
#     new_subgraph_dmrs_nxDG.remove_nodes_from(subgraph_nodes)
#     return True

# def get_subgraphs2nodes_dmrs(dmrs_nxDG,
#                              equalanchor_subgraph_canon,
#                              equalanchor_subgraphs_semicanon,
#                              max_subgraph_size):
#     subgraph_dmrs_nxDG = dmrs_nxDG.copy()
#     size2cnt = Counter()
#     new_all_nodes = set(subgraph_dmrs_nxDG.nodes())
#     all_sizei_nodesSubgraph = gen_all_maxsizek_subgraph(subgraph_dmrs_nxDG,
#                                                         max_subgraph_size,
#                                                         subgraphs_semicanon = equalanchor_subgraphs_semicanon,
#                                                         extract_surfacemap=False)  
#     # from subgraph of size <max_subgraph_size> to 2,
#     # turn it into node if it is present in extracted equal anchored subgraph, greedily
# #     print (dmrs_nxDG.nodes(data=True))
#     for i in range(max_subgraph_size,1,-1):
#         if all_sizei_nodesSubgraph[i]:
#             for sorted_nodes, subgraph_canon, _ in all_sizei_nodesSubgraph[i]:
#                 subgraph_nodes = set(sorted_nodes)
#                 if subgraph_nodes.issubset(new_all_nodes) and subgraph_canon in equalanchor_subgraph_canon:
#                     new_all_nodes -= subgraph_nodes
#                     replacing_node = sum(subgraph_nodes)
#                     new_all_nodes.add(replacing_node)
#                     is_equal_anchored = subgraph2node(subgraph_dmrs_nxDG,
#                                                       sorted_nodes,
#                                                       subgraph_nodes,
#                                                       subgraph_canon,
#                                                       replacing_node)
#                     # given knowing the anchoring, how to handle unequal anchored subgraph?
#                     if not is_equal_anchored:
#                         pass
#                     size2cnt[i] += 1
# #     print (dmrs_nxDG.nodes(data=True))
# #     print ()
#     return subgraph_dmrs_nxDG

def get_node_edges(node, dmrs_nxDG):
    out_edges = list(dmrs_nxDG.out_edges(node, data='label'))
    in_edges = list(dmrs_nxDG.in_edges(node, data='label'))
    return out_edges + in_edges

def get_src_ext(edge_lbl, node_typed = False):
    src_ext_str = edge_lbl.split("-src-")[0][1:]
    if node_typed:
        src_ext_str = src_ext_str[:-1]
    return int(src_ext_str)

        
def get_targ_ext(edge_lbl, node_typed = False):
    targ_ext_str = edge_lbl.split("-targ-")[1][1:]
    if node_typed:
        targ_ext_str = targ_ext_str[1:]
    return int(targ_ext_str)
    
        

def assign_extEdgeLbl(srcExt, edge_lbl, targExt, nodeTypeSrc = "", nodeTypeTarg = ""):
    return '#{}{}-src-'.format(str(srcExt), nodeTypeSrc) + edge_lbl + "-targ-{}#{}".format(nodeTypeTarg, str(targExt))


def replace_subgraph(dmrs_nxDG_repl, dmrs_nxDG, subgraph_nodes_repl, ordered_subgraphsNodes_list,
                     subgraphNodes_isOrdered_list, derivRule, anchors, replacing_node, mode,
                     order_char = False, derivRule_usp = None, node_typed = False):
    # This method is for subgraph replacement to a node with ordered edges (external node) 
    # At the same time, edges interaction btwn subgraphs are extracted and returned
    if not derivRule_usp: derivRule_usp = derivRule
    interSubgrs_edges_lbls = None
    interSubgrs_edges_key_orig = None
    all_node_ind_subgraphs = [dmrs_nxDG.subgraph(subgraphNodes)
                               for subgraphNodes in ordered_subgraphsNodes_list]
    # assume weakly connected
    node_ind_subgraph_repl = dmrs_nxDG_repl.subgraph(subgraph_nodes_repl).copy()
    # extract subgraphs interaction
    if mode in ['UP', 'BP', 'B']:
        if all([True if len(subgraph.nodes) == 0 else is_weakly_connected(subgraph) 
                for subgraph in all_node_ind_subgraphs]):
            if mode in ['UP', 'B']:
                CULR_edges_lbl = list(zip(*get_interSubgr_edges(node_ind_subgraph_repl,
                                                                  ordered_subgraphsNodes_list[0],
                                                                  ordered_subgraphsNodes_list[1])[1]))
                
                UCLR_edges_lbl = list(zip(*get_interSubgr_edges(node_ind_subgraph_repl,
                                                                  ordered_subgraphsNodes_list[1],
                                                                  ordered_subgraphsNodes_list[0])[1]))
#                 if "sb-hd_mc_c" in derivRule:
#                     print (CULR_edges_lbl, UCLR_edges_lbl)
#                 if "vp-vp_crd-fin" in derivRule:
#                     print (CULR_edges_lbl, UCLR_edges_lbl)
                interSubgrs_edges_key_orig = [[], []]
                if CULR_edges_lbl or UCLR_edges_lbl:
                    for idx, edges_lbl in enumerate([CULR_edges_lbl, UCLR_edges_lbl]):
                        if not edges_lbl: continue
                        src_repl = edges_lbl[0][0]
                        lbl = edges_lbl[2][0]
                        src_ext = get_src_ext(lbl, node_typed)
#                         print (src_ext)
                        src_ordered_ext_nodes = dmrs_nxDG_repl.nodes[src_repl]['ordered_ext_nodes'].split("&&")
#                         print (src_ordered_ext_nodes, src_ext)
                        src = src_ordered_ext_nodes[src_ext]
                        targ_repl = edges_lbl[1][0]
                        targ_ext = get_targ_ext(lbl, node_typed)
#                         print (targ_ext)
                        targ_ordered_ext_nodes = dmrs_nxDG_repl.nodes[targ_repl]['ordered_ext_nodes'].split("&&")
#                         print (targ_ordered_ext_nodes, targ_ext)
                        targ = targ_ordered_ext_nodes[targ_ext]
                        interSubgrs_edges_key_orig[idx].append(((src, targ)))
                    interSubgrs_edges_lbls = tuple(() if not edges_lbl else tuple(sorted(edges_lbl[2]))
                        for edges_lbl in [CULR_edges_lbl, UCLR_edges_lbl])
                else:
                    # ignore the extraction if the two subgraph is disconnected
                    pass
#                     print (subgraph_nodes_repl, ordered_subgraphsNodes_list)
#                     write_figs_err(dmrs_nxDG_repl, None, derivRule, derivRule.translate(str.maketrans('', '', string.punctuation)))
#                     input()
            elif mode in ['BP']:
                CL_edges_lbl = get_interSubgr_edges(node_ind_subgraph_repl,
                                                                  ordered_subgraphsNodes_list[0],
                                                                  ordered_subgraphsNodes_list[1])[1]
                LC_edges_lbl = get_interSubgr_edges(node_ind_subgraph_repl,
                                                                  ordered_subgraphsNodes_list[1],
                                                                  ordered_subgraphsNodes_list[0])[1]
                CR_edges_lbl = get_interSubgr_edges(node_ind_subgraph_repl,
                                                                  ordered_subgraphsNodes_list[0],
                                                                  ordered_subgraphsNodes_list[2])[1]
                RC_edges_lbl = get_interSubgr_edges(node_ind_subgraph_repl,
                                                                  ordered_subgraphsNodes_list[2],
                                                                  ordered_subgraphsNodes_list[0])[1]
                LR_edges_lbl = list(zip(*get_interSubgr_edges(node_ind_subgraph_repl,
                                                                  ordered_subgraphsNodes_list[1],
                                                                  ordered_subgraphsNodes_list[2])[1]))
                RL_edges_lbl = list(zip(*get_interSubgr_edges(node_ind_subgraph_repl,
                                                                  ordered_subgraphsNodes_list[2],
                                                                  ordered_subgraphsNodes_list[1])[1]))
                sorted_ccont_nodes = ordered_subgraphsNodes_list[0]
                if len(sorted_ccont_nodes) > 1:
                    edges_lbls_new = [[], [], [], []]
                    for idx, edges_lbl in enumerate([CL_edges_lbl, CR_edges_lbl]):
                        for src, targ, lbl_ext in edges_lbl:
                            src_pos = ""
                            if node_typed:
                                src_pos = lbl_ext.split("-src-")[0][-1]
                            lbl_new = '#{}{}-src-'.format(str(sorted_ccont_nodes.index(src)), src_pos)\
                                + lbl_ext.split("-src-")[1]
                            edges_lbls_new[idx].append((src, targ, lbl_new))
                    for idx, edges_lbl in enumerate([LC_edges_lbl, RC_edges_lbl]):
                        for src, targ, lbl_ext in edges_lbl:
                            targ_pos = ""
                            if node_typed:
                                targ_pos = lbl_ext.split("-targ-")[1][0]
                            lbl_new = lbl_ext.split("-targ-")[0]\
                                + "-targ-{}#{}".format(targ_pos, str(sorted_ccont_nodes.index(targ)))
                            edges_lbls_new[idx + 2].append((src, targ, lbl_new))
                    CL_edges_lbl, CR_edges_lbl, LC_edges_lbl, RC_edges_lbl = edges_lbls_new
                CL_edges_lbl, CR_edges_lbl, LC_edges_lbl, RC_edges_lbl = [list(zip(*l))
                                                                          for l in [CL_edges_lbl, CR_edges_lbl,
                                                                                    LC_edges_lbl, RC_edges_lbl]]
#                 print (CL_edges_lbl)
#                 print (CR_edges_lbl)
#                 print (LC_edges_lbl)
#                 print (RC_edges_lbl)
#                 input()
                interSubgrs_edges_key_orig = [[], [], [], [], [], []]
                for idx, edges_lbl in enumerate([CL_edges_lbl, LC_edges_lbl, CR_edges_lbl,
                                          RC_edges_lbl, LR_edges_lbl, RL_edges_lbl]):
                    if not edges_lbl: continue
                    src_repl = edges_lbl[0][0]
                    lbl = edges_lbl[2][0]
                    src_ext = get_src_ext(lbl, node_typed)
                    if idx in [0, 2] and len(sorted_ccont_nodes) > 1:
                        src = int(dmrs_nxDG_repl.nodes[sorted_ccont_nodes[src_ext]]['ordered_ext_nodes'])
                    else:
                        src_ordered_ext_nodes = dmrs_nxDG_repl.nodes[src_repl]['ordered_ext_nodes'].split("&&")
                        src = src_ordered_ext_nodes[src_ext]
                    targ_repl = edges_lbl[1][0]
                    targ_ext = get_targ_ext(lbl, node_typed)
                    if idx in [1, 3] and len(sorted_ccont_nodes) > 1:
                        targ = int(dmrs_nxDG_repl.nodes[sorted_ccont_nodes[targ_ext]]['ordered_ext_nodes'])
                    else:
                        targ_ordered_ext_nodes = dmrs_nxDG_repl.nodes[targ_repl]['ordered_ext_nodes'].split("&&")
                        targ = targ_ordered_ext_nodes[targ_ext]
                    interSubgrs_edges_key_orig[idx].append(((src, targ)))
                    
#                 if "vp-vp_crd-fin" in derivRule:
#                     print (CL_edges_lbl, LC_edges_lbl, CR_edges_lbl, RC_edges_lbl, LR_edges_lbl, RL_edges_lbl)
                try:
                    interSubgrs_edges_lbls = tuple(() if not edges_lbl else tuple(sorted(edges_lbl[2]))
                            for edges_lbl in [CL_edges_lbl, LC_edges_lbl, CR_edges_lbl, RC_edges_lbl, LR_edges_lbl, RL_edges_lbl])
                except:
                    # print ("")
                    print (CL_edges_lbl, LC_edges_lbl, CR_edges_lbl, RC_edges_lbl, LR_edges_lbl, RL_edges_lbl)
                    # input()
        else:
#             print (derivRule, ": not all subgraphs connected: sem-comp not extracted")
            pass
                
    # obtain external nodes (w/ respect to original dmrs)
    ordered_ext_nodes = get_ordered_external_nodes_repl(node_ind_subgraph_repl,
                                                        ordered_subgraphsNodes_list,
                                                        dmrs_nxDG_repl, node_typed) 
#     print (ordered_ext_nodes)
    ext_node2node_repl = dict()
    for subgraphNodes in ordered_subgraphsNodes_list:
        for node_repl in subgraphNodes:
            orig_external_nodes = dmrs_nxDG_repl.nodes[node_repl]['ordered_ext_nodes'].split("&&")
            for orig_ext_node in orig_external_nodes:
                if orig_ext_node in ordered_ext_nodes:
                    ext_node2node_repl[orig_ext_node] = node_repl
    # obtain subgraph-wise external node order and
    # order the external nodes according to ordered_nodes_list
    # subgraphNodes_isOrdered_list: each element indicates whether the corresponding subgraph nodes supplied are ordered

    # add the replacing node with relevant attr's
    lnk = "<?:?>"
    if anchors:
        lnk = anchors2anchorstr(anchors)
    dmrs_nxDG_repl.add_node(replacing_node,
                            replaced_nodes_repl = "&&".join([str(node) for node in subgraph_nodes_repl]),
                            ordered_ext_nodes = "&&".join(ordered_ext_nodes),
                            derivRule = derivRule,
                            derivRule_usp = derivRule_usp,
                            lnk = lnk)
    if order_char:
        new_order_char = "".join(sorted([dmrs_nxDG_repl.nodes[node]["order_char"]
                                         for nodes in ordered_subgraphsNodes_list
                                             for node in nodes]))
        dmrs_nxDG_repl.nodes[replacing_node]['order_char'] = new_order_char
    # edge information: remove last ext node idx (if any) and add back the latest one
    # ext node idx and lbl tgt determine whether the edge is pointing to the right external node

    new_outedges = []
    for node in ordered_ext_nodes:
        for src,targ,lbl in dmrs_nxDG_repl.out_edges(ext_node2node_repl[node], data='label'):
            if get_src_ext(lbl, node_typed) == dmrs_nxDG_repl.nodes[ext_node2node_repl[node]]\
                                       ["ordered_ext_nodes"].split("&&").index(str(node)):
                src_pos = ""
                if node_typed:
                    src_pos = lbl.split("-src-")[0][-1]
                new_outedges.append((replacing_node, targ,
                     {'label': '#{}{}-src-'.format(str(ordered_ext_nodes.index(node)), src_pos) + lbl.split("-src-")[1]}))

    new_inedges = []
    for node in ordered_ext_nodes:
        for src,targ,lbl in dmrs_nxDG_repl.in_edges(ext_node2node_repl[node], data='label'):
            if get_targ_ext(lbl, node_typed) == dmrs_nxDG_repl.nodes[ext_node2node_repl[node]]\
                                       ["ordered_ext_nodes"].split("&&").index(str(node)):
                targ_pos = ""
                if node_typed:
                    targ_pos = lbl.split("-targ-")[1][0]
                new_inedges.append((src, replacing_node,
                    {'label': lbl.split("-targ-")[0] + "-targ-{}#{}".format(targ_pos, str(ordered_ext_nodes.index(node)))}))

#     print (new_outedges, new_inedges)
    # replace subgraph with node
    dmrs_nxDG_repl.add_edges_from(new_outedges)
    dmrs_nxDG_repl.add_edges_from(new_inedges)
    dmrs_nxDG_repl.remove_nodes_from(subgraph_nodes_repl)
#     print (dmrs_nxDG_repl.out_edges(data = 'label'))
#     if replacing_node == 64100:
#         pprint (new_outedges)
#         pprint (dmrs_nxDG_repl.edges)
#     print (mode)
#     print (derivRule)
#     print (derivRule_usp)
#     write_figs_err(dmrs_nxDG_repl, None, derivRule, str(derivRule).replace("/", "_")[:50])
#         input()
    # record subgraph 2 deriv. rule
#     node_ind_subgraph = dmrs_nxDG_repl.subgraph(subgraph_nodes_repl)
#     for idx, ext_node in enumerate(ext_nodes): 
#         node_ind_subgraph.nodes[ext_node]['ext_idx'] = idx
    # node_ind_subgraph shd already include the edge info for successful unification (ext node idx + edge lbl)
    # node replacing a node is not recorded
#         if 
#         ordered_subgraphsNodes_list[0]
#             for subgraphNodes in ordered_subgraphsNodes_list:
                
#             get_interSubgr_edges(node_ind_subgraph_repl, )
#         subgraph2derivRule = (node_ind_subgraph_repl, derivRule)
        
    # uncomment this line if you wish to visualize the replacement process
#     write_figs_err(dmrs_nxDG_repl, None, derivRule, derivRule.translate(str.maketrans('', '', string.punctuation)))
#     write_figs_err(node_ind_subgraph_repl, None, derivRule, derivRule.translate(str.maketrans('', '', string.punctuation)))
#     input()
    return interSubgrs_edges_lbls, interSubgrs_edges_key_orig

                                  

def get_interSubgr_edges(dmrs_nxDG, node_set1, node_set2, directed = True):
    # return keys and labels of multiedge
    if node_set2 == None:
        interSubgr_edges_1to2 = list(edge_boundary(dmrs_nxDG,
                                          nbunch1 = node_set1,
                                          keys = True,
                                          data = 'label'))
    else:
        interSubgr_edges_1to2 = list(edge_boundary(dmrs_nxDG,
                                              nbunch1 = node_set1,
                                              nbunch2 = node_set2,
                                              keys = True,
                                              data = 'label'))
    interSubgr_edges = interSubgr_edges_1to2
    if not directed:
        if node_set1 == None:
            interSubgr_edges_2to1 = list(edge_boundary(dmrs_nxDG,
                                                nbunch1 = node_set2,
                                                keys = True,
                                                data = 'label'))
        else:
            interSubgr_edges_2to1 = list(edge_boundary(dmrs_nxDG,
                                                nbunch1 = node_set2,
                                                nbunch2 = node_set1,
                                                keys = True,
                                                data = 'label'))
        interSubgr_edges += interSubgr_edges_2to1
    interSubgr_edges_key = [(t[0],t[1],t[2]) for t in interSubgr_edges]
    interSubgr_edges_lbl = [(t[0],t[1],t[3]) for t in interSubgr_edges]
    return interSubgr_edges_key, interSubgr_edges_lbl

def get_edgeLbl_from_extEdgeLbl(extEdge_lbl):
    return extEdge_lbl.split("-src-")[1].split("-targ-")[0]

twoIdx2selfIdx = (0, 2)
twoIdx2nb = (1, 0)
twoIdx2nbIdx = (3, 1)
threeIdx2selfIdx = (0, 2, 0, 4, 2, 4)
threeIdx2nb = (1, 0, 2, 0, 2, 1)
threeIdx2nbIdx = (3, 1, 5, 1, 5, 3)

def find_int_edges_lbl2(dgtrs_extEdges, dgtrs_extEdges_dicts, bitVecs, edgeKey2lbl, dmrs_node2bitVec, node2pos = None, node_typed = False, p = False):
    global twoIdx2selfIdx, twoIdx2nb, twoIdx2nbIdx, threeIdx2selfIdx, threeIdx2nb, threeIdx2nbIdx
#    try:
#     print (dmrs_node2bitVec, bitVecs)

    if node_typed:
        interSubgrsEdges = tuple(frozenset(
                ['#{}{}-src-{}-targ-{}#{}'.format(ext, node2pos[s], lbl, node2pos[t], dgtrs_extEdges_dicts[twoIdx2nbIdx[idx]][(s,t,k)][1])
                 for (s,t,k), (st, ext, lbl) in dgtrs_extEdges_dicts[twoIdx2selfIdx[idx]].items()
                 if st == 's' and dmrs_node2bitVec[t] & bitVecs[nb] == dmrs_node2bitVec[t]
                 ])
            for idx, nb in enumerate(twoIdx2nb))
    #    except Exception as e:
            #if not isinstance(e, timeout.TimeoutException):
                #traceback.print_exc()
                #pprint (dgtrs_extEdges)
                #print ("--")
                #pprint (dgtrs_extEdges_dicts)
                #print ("----")
                #print (bin(bitVecs[0]), bin(bitVecs[1]))
                #input()
        interSubgrsEdges_keys = tuple(
                [(s,t,k) for (s,t,k), (st, ext, lbl) in dgtrs_extEdges_dicts[twoIdx2selfIdx[idx]].items()
                 if st == 's' and dmrs_node2bitVec[t] & bitVecs[nb] == dmrs_node2bitVec[t]
                 ]
            for idx, nb in enumerate(twoIdx2nb))
        
    else:
        interSubgrsEdges = tuple(frozenset(
                ['#{}-src-{}-targ-#{}'.format(ext, lbl, dgtrs_extEdges_dicts[twoIdx2nbIdx[idx]][(s,t,k)][1])
                 for (s,t,k), (st, ext, lbl) in dgtrs_extEdges_dicts[twoIdx2selfIdx[idx]].items()
                 if st == 's' and dmrs_node2bitVec[t] & bitVecs[nb] == dmrs_node2bitVec[t]
                 ])
            for idx, nb in enumerate(twoIdx2nb))
    #    except Exception as e:
            #if not isinstance(e, timeout.TimeoutException):
                #traceback.print_exc()
                #pprint (dgtrs_extEdges)
                #print ("--")
                #pprint (dgtrs_extEdges_dicts)
                #print ("----")
                #print (bin(bitVecs[0]), bin(bitVecs[1]))
                #input()
        interSubgrsEdges_keys = tuple(
                [(s,t,k) for (s,t,k), (st, ext, lbl) in dgtrs_extEdges_dicts[twoIdx2selfIdx[idx]].items()
                 if st == 's' and dmrs_node2bitVec[t] & bitVecs[nb] == dmrs_node2bitVec[t]
                 ]
            for idx, nb in enumerate(twoIdx2nb))
    return interSubgrsEdges, interSubgrsEdges_keys

def find_int_edges_lbl3(dgtrs_extEdges, dgtrs_extEdges_dicts, bitVecs, edgeKey2lbl, dmrs_node2bitVec, node2pos = None, node_typed = False, p = False):
    global twoIdx2selfIdx, twoIdx2nb, twoIdx2nbIdx, threeIdx2selfIdx, threeIdx2nb, threeIdx2nbIdx
    if node_typed:
        interSubgrsEdges = tuple(frozenset(
            ['#{}{}-src-{}-targ-{}#{}'.format(ext, node2pos[s], lbl, node2pos[t], dgtrs_extEdges_dicts[threeIdx2nbIdx[idx]][(s,t,k)][1])
             for (s,t,k), (st, ext, lbl) in dgtrs_extEdges_dicts[threeIdx2selfIdx[idx]].items()
             if st == 's' and bitVecs[nb] and dmrs_node2bitVec[t] & bitVecs[nb] == dmrs_node2bitVec[t]\
             and (s,t,k) in dgtrs_extEdges_dicts[threeIdx2nbIdx[idx]]
             ])
        for idx, nb in enumerate(threeIdx2nb))
        interSubgrsEdges_keys = tuple(
            [(s,t,k)
             for (s,t,k), (st, ext, lbl) in dgtrs_extEdges_dicts[threeIdx2selfIdx[idx]].items()
             if st == 's' and bitVecs[nb] and dmrs_node2bitVec[t] & bitVecs[nb] == dmrs_node2bitVec[t]\
             and (s,t,k) in dgtrs_extEdges_dicts[threeIdx2nbIdx[idx]]
             ]
        for idx, nb in enumerate(threeIdx2nb))
    else:
        interSubgrsEdges = tuple(frozenset(
            ['#{}-src-{}-targ-#{}'.format(ext, lbl, dgtrs_extEdges_dicts[threeIdx2nbIdx[idx]][(s,t,k)][1])
             for (s,t,k), (st, ext, lbl) in dgtrs_extEdges_dicts[threeIdx2selfIdx[idx]].items()
             if st == 's' and bitVecs[nb] and dmrs_node2bitVec[t] & bitVecs[nb] == dmrs_node2bitVec[t]\
             and (s,t,k) in dgtrs_extEdges_dicts[threeIdx2nbIdx[idx]]
             ])
        for idx, nb in enumerate(threeIdx2nb))
        interSubgrsEdges_keys = tuple(
            [(s,t,k)
             for (s,t,k), (st, ext, lbl) in dgtrs_extEdges_dicts[threeIdx2selfIdx[idx]].items()
             if st == 's' and bitVecs[nb] and dmrs_node2bitVec[t] & bitVecs[nb] == dmrs_node2bitVec[t]\
             and (s,t,k) in dgtrs_extEdges_dicts[threeIdx2nbIdx[idx]]
             ]
        for idx, nb in enumerate(threeIdx2nb))
    return interSubgrsEdges, interSubgrsEdges_keys

def find_extEdgeKeyExts_list(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl):
    commonEdgeKeys_cnter = Counter([edge for dgtrs_extEdges_dict in dgtrs_extEdges_dicts for edge in dgtrs_extEdges_dict])
    extEdgeKeyExts_list = [[(key, ext) for key, ext in dgtrs_extEdge if commonEdgeKeys_cnter[key] == 1] for dgtrs_extEdge in dgtrs_extEdges]
    return extEdgeKeyExts_list
    
def find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl, extEdgeKeyExts_list = None):
    extEdges_new = []
    extEdges_dicts_new = ({}, {})
    push_item2extEdges_new = extEdges_new.append
    if not extEdgeKeyExts_list:
        commonEdgeKeys_cnter = Counter([edge for dgtrs_extEdges_dict in dgtrs_extEdges_dicts for edge in dgtrs_extEdges_dict])
        extEdgeKeyExts_list = [[(key, ext) for key, ext in dgtrs_extEdge if commonEdgeKeys_cnter[key] == 1] for dgtrs_extEdge in dgtrs_extEdges]
    ext_node_idx = -1
    st2idx = {'s': 0, 't': 1}
    for idx, keyExt_list in enumerate(extEdgeKeyExts_list):
        prev_old_ext = -1
        for key, (srcOrTarg, old_ext) in keyExt_list:
            if prev_old_ext != old_ext:
                ext_node_idx += 1
                prev_old_ext = old_ext
            push_item2extEdges_new((key,(srcOrTarg, ext_node_idx)))
            extEdges_dicts_new[st2idx[(srcOrTarg)]][key] = (srcOrTarg, ext_node_idx, edgeKey2lbl[key])
    return tuple(extEdges_new), extEdges_dicts_new, extEdgeKeyExts_list
                
# def find_intExt_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, bitVecs, edgeKey2lbl, dmrs_node2bitVec, findInt = True, findExt = False, p = False):
#     global twoIdx2selfIdx, twoIdx2nb, twoIdx2nbIdx, threeIdx2selfIdx, threeIdx2nb, threeIdx2nbIdx
#     extEdges_new = []
#     extEdges_dicts_new = [{}, {}]
#     interSubgrsEdges = ()
#     if findInt:
#         len_dgtrs_extEdges = len(dgtrs_extEdges)
#         if len_dgtrs_extEdges == 3:
#             interSubgrsEdges = tuple(frozenset(
#                 ['#{}-src-{}-targ-#{}'.format(ext, lbl, dgtrs_extEdges_dicts[threeIdx2nbIdx[idx]][(s,t,k)][1])
#                  for (s,t,k), (st, ext, lbl) in dgtrs_extEdges_dicts[threeIdx2selfIdx[idx]].items()
#                  if st == 's' and bitVecs[nb] and dmrs_node2bitVec[t] & bitVecs[nb] == dmrs_node2bitVec[t]\
#                  and (s,t,k) in dgtrs_extEdges_dicts[threeIdx2nbIdx[idx]]
#                  ])
#             for idx, nb in enumerate(threeIdx2nb))
#         elif len_dgtrs_extEdges == 2:
#             interSubgrsEdges = tuple(frozenset(
#                 ['#{}-src-{}-targ-#{}'.format(ext, lbl, dgtrs_extEdges_dicts[twoIdx2nbIdx[idx]][(s,t,k)][1])
#                  for (s,t,k), (st, ext, lbl) in dgtrs_extEdges_dicts[twoIdx2selfIdx[idx]].items()
#                  if st == 's' and dmrs_node2bitVec[t] & bitVecs[nb] == dmrs_node2bitVec[t]
#                  ]) 
#             for idx, nb in enumerate(twoIdx2nb))
#     if findExt:
#         push_item2extEdges_new = extEdges_new.append
#         commonEdgeKeys_cnter = Counter([edge for dgtrs_extEdges_dict in dgtrs_extEdges_dicts for edge in dgtrs_extEdges_dict])
#         extEdgeKeyExts_list = [[(key, ext) for key, ext in dgtrs_extEdge if commonEdgeKeys_cnter[key] == 1] for dgtrs_extEdge in dgtrs_extEdges]
#         ext_node_idx = -1
#         st2idx = {'s': 0, 't': 1}
#         for idx, keyExt_list in enumerate(extEdgeKeyExts_list):
#             prev_old_ext = -1
#             for key, (srcOrTarg, old_ext) in keyExt_list:
#                 if prev_old_ext != old_ext:
#                     ext_node_idx += 1
#                     prev_old_ext = old_ext
#                 push_item2extEdges_new((key,(srcOrTarg, ext_node_idx)))
#                 extEdges_dicts_new[st2idx[(srcOrTarg)]][key] = (srcOrTarg, ext_node_idx, edgeKey2lbl[key])
#     return tuple(extEdges_new), extEdges_dicts_new, interSubgrsEdges

# mode2func = {'U': matched_semComp_U, 'B': matched_semComp_B, 'BPSemEmt': matched_semComp_BPSemEmt,
#              'BP': matched_semComp_BP}


def matched_semComp(dmrs_nxDG_ext, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts,
                         copTrgs, semEmtTrgs, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items, mode, p = False):
    data_new = None
    if mode == 'U':
        shrg_rules = SHRG.get(dgtr_derivRules)
        return shrg_rules, data_new
        
    if mode == 'B':
        dgtrs_extEdges_dicts = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1])
        interSubgrsEdges_got = find_int_edges_lbl2(dgtrs_extEdges, dgtrs_extEdges_dicts, bitVecs, edgeKey2lbl, 
                                                   dmrs_node2bitVec)
        shrg_key = (dgtr_derivRules, interSubgrsEdges_got)
        shrg_rules = SHRG.get(shrg_key)
        if shrg_rules:
            extEdges_new, extEdges_dicts_new = find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl)
            data_new = (extEdges_new, extEdges_dicts_new)
        return shrg_rules, data_new
    
    elif mode == 'SubgrTag':
        # update new ext edge for the wsubgraph
        out_edges_list = [list(dmrs_nxDG_ext.out_edges(dgtr_node, keys = True, data = 'label')) for dgtr_node in dgtrs_nodes]
        in_edges_list = [list(dmrs_nxDG_ext.in_edges(dgtr_node, keys = True, data = 'label')) for dgtr_node in dgtrs_nodes]
#         print (list(zip(out_edges_list, in_edges_list)))
        dgtrs_extEdges_out = [{(s,t,k): ("s", int(l.split("-src-")[0][1:]), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in out_edges} for out_edges in out_edges_list]
        dgtrs_extEdges_in = [{(s,t,k): ("t", int(l.split("-targ-")[1][1:]), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in in_edges} for in_edges in in_edges_list]
        dgtrs_extEdges_dicts = []
        for idx, item in enumerate(dgtrs_extEdges_out):
            dgtrs_extEdges_dicts.append(item)
            dgtrs_extEdges_dicts.append(dgtrs_extEdges_in[idx])
#         print (dgtrs_extEdges_dicts)
    
        dgtrs_extEdges = tuple(tuple(sorted([((s,t,k), ("s", int(l.split("-src-")[0][1:]))) if (s,t,k,l) in out_edges
                                                   else ((s,t,k),("t", int(l.split("-targ-")[1][1:])))
                                               for s,t,k,l in out_edges + in_edges], key = lambda x: x[1][1])
                                     )
                                for out_edges, in_edges in list(zip(out_edges_list, in_edges_list)))
        new_extEdge, new_extEdges_dicts = find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl)
#         print (new_extEdges_dicts)
        return new_extEdge, tuple(new_extEdges_dicts)
    
    elif mode == 'Tag':
        out_edges = list(dmrs_nxDG_ext.out_edges(dgtrs_nodes[0], keys = True, data = 'label'))
        in_edges = list(dmrs_nxDG_ext.in_edges(dgtrs_nodes[0], keys = True, data = 'label'))
        dgtrs_extEdges_out = {(s,t,k): ("s", int(l.split("-src-")[0][1:]), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in out_edges}
#         print (in_edges)
        dgtrs_extEdges_in = {(s,t,k): ("t", int(l.split("-targ-")[1][1:]), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in in_edges}
        dgtrs_extEdges_dicts = []
        dgtrs_extEdges_dicts.append(dgtrs_extEdges_out)
        dgtrs_extEdges_dicts.append(dgtrs_extEdges_in)
#         print (dgtrs_extEdges_dicts)
        dgtrs_extEdges = (tuple(sorted([((s,t,k), ("s", int(l.split("-src-")[0][1:]))) if (s,t,k,l) in out_edges
                                             else ((s,t,k),("t", int(l.split("-targ-")[1][1:])))
                                         for s,t,k,l in out_edges + in_edges], key = lambda x: x[1][1])),)
        new_extEdge, new_extEdges_dicts = find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl)
#         print (dgtrs_nodes, new_extEdge)
        return new_extEdge, tuple(new_extEdges_dicts)

def matched_semComp_semEmt(dmrs_nxDG_ext, dgtr_derivRules_exact, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts,
                         semEmtTrgs, scopes, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items,
                         mode, node2pos = None, node_typed = False, p = False, directed = False, return_exts = False):
    
    shrgKey2psvItem = defaultdict()
    try:
        semEmtTrgs_T = tuple(zip(*semEmtTrgs))
    except:
        print (semEmtTrgs)
        input()
#     if mode in ['predSemEmt']:
#         if no_items == 3:
#             pass
        
    if mode in ['copula', 'prtcl', 'compl', 'by']:
        derivRule2psvItem = defaultdict(list)
        dgtrsTrg2edges2surf2logProb = SHRG
        interSubgrsEdges_gotA = None
        interSubgrsEdges_gotB = None
        extEdgeKeyExts_listA = None
        extEdgeKeyExts_listB = None
        if no_items == 3:
            dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[2])
        elif no_items == 2:
            dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1])
        for d1, d2 in [(0,1), (2,3)]:
            if d1 == 0: 
                if no_items == 3:
                    dgtr_derivRules_init = (dgtr_derivRules[0], dgtr_derivRules[1], (dgtr_derivRules[2][0], None))
                elif no_items == 2:
                    dgtr_derivRules_init = (dgtr_derivRules[0], (dgtr_derivRules[1][0], None))
            elif d1 == 2:
                if no_items == 3:
                    dgtr_derivRules_init = (dgtr_derivRules[0], (dgtr_derivRules[1][0], None), dgtr_derivRules[2])
                elif no_items == 2:
                    dgtr_derivRules_init = ((dgtr_derivRules[0][0], None), dgtr_derivRules[1])
            if directed[d1]:
                edges2derivRuleSurf2logProbA = dgtrsTrg2edges2surf2logProb.get(dgtr_derivRules_init)
#                 print (dgtr_derivRules_init, edges2derivRuleSurf2logProbA, "!")
                if edges2derivRuleSurf2logProbA:
                    if not interSubgrsEdges_gotA:
                        if no_items == 2:
                            interSubgrsEdges_gotA, interSubgrsEdges_keysA,  = find_int_edges_lbl2(dgtrs_extEdges,
                                                                       dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                                         dmrs_node2bitVec, node2pos
, node_typed)
                        elif no_items == 3:
                            interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl3(dgtrs_extEdges,
                                                                       dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                                         dmrs_node2bitVec, node2pos
, node_typed)
                    derivRuleSurf2logProbA = edges2derivRuleSurf2logProbA.get(interSubgrsEdges_gotA)
#                     print (dgtr_derivRules_init, interSubgrsEdges_gotA, derivRuleSurf2logProbA)
                    if derivRuleSurf2logProbA:
                        if not extEdgeKeyExts_listA:
                            extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA\
                                = find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dictsA, edgeKey2lbl)
                        for (derivRule_new, surf), logProb in derivRuleSurf2logProbA.items():
                            trgs_newA = [None, None, None]
                            for idx, semEmt_type in enumerate(['copula', 'prtcl', 'compl']):
#                                 if semEmt_type == mode: trgs_newA[idx] = None
                                trgs_newA[idx] = get_dgtrsTrg_test((semEmtTrgs[-2][idx], semEmtTrgs[-1][idx], ), derivRule_new)
                            trgs_newA = tuple(trgs_newA)
#                             print (trgs_newA)
                            derivRule2psvItem[(derivRule_new, interSubgrsEdges_gotA)].append((bitVecs,
                                                                            dgtr_derivRules_exact,
                                                                    trgs_newA, scopes, logProb,
                                                                     surf, extEdges_newA, extEdges_dicts_newA,
                                                                     dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA))
            if directed[d2]:
                if no_items == 3:
                    edges2derivRuleSurf2logProbB = dgtrsTrg2edges2surf2logProb.get((dgtr_derivRules_init[0],
                                                                                    dgtr_derivRules_init[2],
                                                                                dgtr_derivRules_init[1]))
                elif no_items == 2:
                    edges2derivRuleSurf2logProbB = dgtrsTrg2edges2surf2logProb.get((dgtr_derivRules_init[1],
                                                                                dgtr_derivRules_init[0]))
                if edges2derivRuleSurf2logProbB:
            #         print (dgtr_derivRules, edges2derivRuleSurf2logProbA, edges2derivRuleSurf2logProbB)
                    if not interSubgrsEdges_gotA:
                        if no_items == 2:
                            interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl2(dgtrs_extEdges,
                                                                dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                                         dmrs_node2bitVec,  node2pos
, node_typed)
                        elif no_items == 3:
                            interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl3(dgtrs_extEdges,
                                                                dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                                         dmrs_node2bitVec, node2pos
, node_typed)
                    if no_items == 3:
                        interSubgrsEdges_gotB = (interSubgrsEdges_gotA[2], interSubgrsEdges_gotA[3],
                                                 interSubgrsEdges_gotA[0], interSubgrsEdges_gotA[1],
                                                 interSubgrsEdges_gotA[5], interSubgrsEdges_gotA[4])
                        interSubgrsEdges_keysB = (interSubgrsEdges_keysA[2], interSubgrsEdges_keysA[3],
                                                 interSubgrsEdges_keysA[0], interSubgrsEdges_keysA[1],
                                                 interSubgrsEdges_keysA[5], interSubgrsEdges_keysA[4])
                    elif no_items == 2:
                        interSubgrsEdges_gotB = (interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[0])
                        interSubgrsEdges_keysB = (interSubgrsEdges_keysA[1], interSubgrsEdges_keysA[0])
                    derivRuleSurf2logProbB = edges2derivRuleSurf2logProbB.get(interSubgrsEdges_gotB)
#                     print ((dgtr_derivRules_init[1], dgtr_derivRules_init[0]), interSubgrsEdges_gotB, derivRuleSurf2logProbB)
                    if derivRuleSurf2logProbB:
                        if no_items == 3:
                
                            dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[2],
                                                     *dgtrs_extEdges_dicts[1])
                            dgtrs_extEdgesB = (dgtrs_extEdges[0], dgtrs_extEdges[2], dgtrs_extEdges[1])
                            dgtr_derivRules_exactB = (dgtr_derivRules_exact[0], dgtr_derivRules_exact[2], 
                                                dgtr_derivRules_exact[1])
                            bitVecsB = (bitVecs[0], bitVecs[2], bitVecs[1])
                            scopesB = (scopes[0], scopes[2], scopes[1])
                        elif no_items == 2:
                            dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[0])
                            dgtrs_extEdgesB = (dgtrs_extEdges[1], dgtrs_extEdges[0])
                            dgtr_derivRules_exactB = (dgtr_derivRules_exact[1], dgtr_derivRules_exact[0])
                            bitVecsB = (bitVecs[1], bitVecs[0])
                            scopesB = (scopes[1], scopes[0])
                        if not extEdgeKeyExts_listB:
                            extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB\
                                = find_ext_edges_lbl(dgtrs_extEdgesB, dgtrs_extEdges_dictsB, edgeKey2lbl)
                        for (derivRule_new, surf), logProb in derivRuleSurf2logProbB.items():
                            trgs_newB = [None, None, None]
                            for idx, semEmt_type in enumerate(['copula', 'prtcl', 'compl']):
#                                 if semEmt_type == mode: trgs_newB[idx] = None
                                trgs_newB[idx] = get_dgtrsTrg_test((semEmtTrgs[-1][idx], semEmtTrgs[-2][idx], ), derivRule_new)
                            trgs_newB = tuple(trgs_newB)
#                             print (trgs_newB, "B")
                            derivRule2psvItem[(derivRule_new, interSubgrsEdges_gotB)].append((bitVecsB,
                                                                        dgtr_derivRules_exactB, trgs_newB,
                                                                            scopesB, logProb, surf, 
                                                                     extEdges_newB, extEdges_dicts_newB,
                                                                    dgtrs_extEdgesB, extEdgeKeyExts_listB, interSubgrsEdges_keysA))

        return derivRule2psvItem
        
                
    
#     elif mode == 'SubgrTag':
#         # update new ext edge for the wsubgraph
#         out_edges_list = [list(dmrs_nxDG_ext.out_edges(dgtr_node, keys = True, data = 'label')) for dgtr_node in dgtrs_nodes]
#         in_edges_list = [list(dmrs_nxDG_ext.in_edges(dgtr_node, keys = True, data = 'label')) for dgtr_node in dgtrs_nodes]
# #         print (list(zip(out_edges_list, in_edges_list)))
#         dgtrs_extEdges_out = [{(s,t,k): ("s", int(l.split("-src-")[0][1:]), edgeKey2lbl[(s,t,k)])
#                                 for s,t,k,l in out_edges} for out_edges in out_edges_list]
#         dgtrs_extEdges_in = [{(s,t,k): ("t", int(l.split("-targ-")[1][1:]), edgeKey2lbl[(s,t,k)])
#                                 for s,t,k,l in in_edges} for in_edges in in_edges_list]
#         dgtrs_extEdges_dicts = []
#         for idx, item in enumerate(dgtrs_extEdges_out):
#             dgtrs_extEdges_dicts.append(item)
#             dgtrs_extEdges_dicts.append(dgtrs_extEdges_in[idx])
# #         print (dgtrs_extEdges_dicts)
    
#         dgtrs_extEdges = tuple(tuple(sorted([((s,t,k), ("s", int(l.split("-src-")[0][1:]))) if (s,t,k,l) in out_edges
#                                                    else ((s,t,k),("t", int(l.split("-targ-")[1][1:])))
#                                                for s,t,k,l in out_edges + in_edges], key = lambda x: x[1][1])
#                                      )
#                                 for out_edges, in_edges in list(zip(out_edges_list, in_edges_list)))
#         new_extEdge, new_extEdges_dicts, extEdgeKeyExts_list\
#             = find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl)
# #         print (new_extEdges_dicts)
#         return new_extEdge, new_extEdges_dicts
    
#     elif mode == 'Tag':
#         out_edges = list(dmrs_nxDG_ext.out_edges(dgtrs_nodes[0], keys = True, data = 'label'))
#         in_edges = list(dmrs_nxDG_ext.in_edges(dgtrs_nodes[0], keys = True, data = 'label'))
#         dgtrs_extEdges_out = {(s,t,k): ("s", int(l.split("-src-")[0][1:]), edgeKey2lbl[(s,t,k)])
#                                 for s,t,k,l in out_edges}
# #         print (in_edges)
#         dgtrs_extEdges_in = {(s,t,k): ("t", int(l.split("-targ-")[1][1:]), edgeKey2lbl[(s,t,k)])
#                                 for s,t,k,l in in_edges}
#         dgtrs_extEdges_dicts = []
#         dgtrs_extEdges_dicts.append(dgtrs_extEdges_out)
#         dgtrs_extEdges_dicts.append(dgtrs_extEdges_in)
# #         print (dgtrs_extEdges_dicts)
#         dgtrs_extEdges = (tuple(sorted([((s,t,k), ("s", int(l.split("-src-")[0][1:]))) if (s,t,k,l) in out_edges
#                                              else ((s,t,k),("t", int(l.split("-targ-")[1][1:])))
#                                          for s,t,k,l in out_edges + in_edges], key = lambda x: x[1][1])),)
#         new_extEdge, new_extEdges_dicts, extEdgeKeyExts_list =\
#             find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl)
# #         print (dgtrs_nodes, new_extEdge)
#         return new_extEdge, new_extEdges_dicts
def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)


def matched_semComp_lazy_ext(dmrs_nxDG_ext, dgtr_derivRules_prob, dgtr_derivRules_exact, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts,
                         dgtrs_trgs, scopes, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items,
                         mode, node2pos = None, node_typed = False, p = False, directed = False, return_exts = False):
    
    shrgKey2psvItem = defaultdict()
    if mode == 'U':
#         print (dgtr_derivRules)
#         print (SHRG.keys())
        if dgtr_derivRules in SHRG:
            # print ("v;", SHRG[dgtr_derivRules])
            shrgKey2psvItem[dgtr_derivRules] = dgtrs_extEdges[0], dgtrs_extEdges_dicts[0]
        return shrgKey2psvItem
    
    elif mode == 'B':
        # dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1])
        # interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl2(dgtrs_extEdges,
        #                                                    dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
        #                                                    dmrs_node2bitVec, node2pos
        # , node_typed)
        # shrg_keyA = (dgtr_derivRules, interSubgrsEdges_gotA)
#         if p:
        # print (shrg_keyA)
        # for k in SHRG:
        #     print (k)
        #     input()
#             print ()
#             print (SHRG)
#             print ()
#             print ()
        if shrg_keyA in SHRG:
            # print (mode, shrg_keyA)
            extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                                     dgtrs_extEdges_dictsA, edgeKey2lbl)
            shrgKey2psvItem[(shrg_keyA, dgtr_derivRules_exact, dgtr_derivRules_prob)]\
                = (bitVecs, dgtrs_trgs, scopes, extEdges_newA, extEdges_dicts_newA,
                                          dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
        if not directed:
            interSubgrsEdges_gotB = (interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[0])
            interSubgrsEdges_keysB = (interSubgrsEdges_keysA[1], interSubgrsEdges_keysA[0])
            shrg_keyB = ((dgtr_derivRules[1], dgtr_derivRules[0]), interSubgrsEdges_gotB)
            if shrg_keyB in SHRG:
                # print (mode, shrg_keyB)
                dgtr_derivRules_probB = (dgtr_derivRules_prob[1], dgtr_derivRules_prob[0])
                dgtr_derivRules_exactB = (dgtr_derivRules_exact[1], dgtr_derivRules_exact[0])
                dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[0])
                dgtrs_extEdgesB = (dgtrs_extEdges[1], dgtrs_extEdges[0])
                bitVecsB = (bitVecs[1], bitVecs[0])
                dgtrs_trgsB = (dgtrs_trgs[1], dgtrs_trgs[0])
                scopesB = (scopes[1], scopes[0])
                extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                         dgtrs_extEdges_dictsB, edgeKey2lbl) 
                shrgKey2psvItem[(shrg_keyB, dgtr_derivRules_exactB, dgtr_derivRules_probB)]\
                    = (bitVecsB, dgtrs_trgsB, scopesB, extEdges_newB,
                                              extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB
                                             , interSubgrsEdges_keysB)
        return shrgKey2psvItem
    
    elif mode == 'B-predSemEmt':
        dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1])
        interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl2(dgtrs_extEdges,
                                                           dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                             dmrs_node2bitVec, node2pos
, node_typed)
        shrg_keyA = interSubgrsEdges_gotA
#         if p:
#             print (shrg_keyA)
#             print ()
#             print (SHRG)
#             print ()
#             print ()
        if shrg_keyA in SHRG[dgtr_derivRules]:
#             print (mode, shrg_keyA)
            extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                                     dgtrs_extEdges_dictsA, edgeKey2lbl)
            shrgKey2psvItem[((dgtr_derivRules, shrg_keyA), dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs,
                                                                                      scopes, extEdges_newA,
                                                             extEdges_dicts_newA,
                                          dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
        if not directed:
            interSubgrsEdges_gotB = (interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[0])
            interSubgrsEdges_keysB = (interSubgrsEdges_keysA[1], interSubgrsEdges_keysA[0])
            shrg_keyB = interSubgrsEdges_gotB
            if shrg_keyB in SHRG[(dgtr_derivRules[1], dgtr_derivRules[0])]:
#             print (mode, shrg_keyB)
                dgtr_derivRules_exactB = (dgtr_derivRules_exact[1], dgtr_derivRules_exact[0])
                dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[0])
                dgtrs_extEdgesB = (dgtrs_extEdges[1], dgtrs_extEdges[0])
                bitVecsB = (bitVecs[1], bitVecs[0])
                dgtrs_trgsB = (dgtrs_trgs[1], dgtrs_trgs[0])
                scopesB = (scopes[1], scopes[0])
                extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                         dgtrs_extEdges_dictsB, edgeKey2lbl) 
                shrgKey2psvItem[(((dgtr_derivRules[1], dgtr_derivRules[0]), shrg_keyB),
                                 dgtr_derivRules_exactB)] = (bitVecsB, dgtrs_trgsB,
                                                                                          scopesB, extEdges_newB,
                                              extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB
                                             , interSubgrsEdges_keysB)
        return shrgKey2psvItem
    
    
    elif mode == 'BP-predSemEmt':
        dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[2])
        interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl3(dgtrs_extEdges,
                                                             dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                             dmrs_node2bitVec, node2pos
, node_typed)
        
        shrg_keyA = interSubgrsEdges_gotA
#         print ()
        if shrg_keyA in SHRG[dgtr_derivRules]:
#             print ("123312")
            extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                             dgtrs_extEdges_dictsA, edgeKey2lbl)
            shrgKey2psvItem[((dgtr_derivRules, shrg_keyA), dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs, scopes, extEdges_newA,
                                                             extEdges_dicts_newA, 
                                          dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
        if not directed:
#         print (shrg_keyA)
            interSubgrsEdges_gotB = (interSubgrsEdges_gotA[2], interSubgrsEdges_gotA[3], interSubgrsEdges_gotA[0],
                                 interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[5], interSubgrsEdges_gotA[4])
            interSubgrsEdges_keysB = (interSubgrsEdges_keysA[2], interSubgrsEdges_keysA[3],
                                      interSubgrsEdges_keysA[0], interSubgrsEdges_keysA[1],
                                      interSubgrsEdges_keysA[5], interSubgrsEdges_keysA[4])
        
            shrg_keyB = interSubgrsEdges_gotB
    #         print (shrg_keyB)

            if shrg_keyB in SHRG[(dgtr_derivRules[0], dgtr_derivRules[2], dgtr_derivRules[1])]:
    #             print (mode, shrg_keyB)
                dgtr_derivRules_exactB = (dgtr_derivRules_exact[0], dgtr_derivRules_exact[2], dgtr_derivRules_exact[1])
                dgtrs_extEdgesB = (dgtrs_extEdges[0], dgtrs_extEdges[2], dgtrs_extEdges[1])
                dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[2], *dgtrs_extEdges_dicts[1])
                bitVecsB = (bitVecs[0], bitVecs[2], bitVecs[1])
                dgtrs_trgsB = (dgtrs_trgs[0], dgtrs_trgs[2], dgtrs_trgs[1])
                scopesB = (scopes[0], scopes[2], scopes[1])
                extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                 dgtrs_extEdges_dictsB, edgeKey2lbl)
                shrgKey2psvItem[(((dgtr_derivRules[0], dgtr_derivRules[2], dgtr_derivRules[1]), shrg_keyB),
                               dgtr_derivRules_exactB)]\
                    = (bitVecsB, dgtrs_trgsB, scopesB, extEdges_newB,
                                              extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB,
                                             interSubgrsEdges_keysB)
        return shrgKey2psvItem    
                
    
    elif mode == 'BPsemEmt':
        extEdges_newA = None
        dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[2])
        interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl3(dgtrs_extEdges,
                                                            dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                             dmrs_node2bitVec, node2pos
, node_typed)
#         _, _, interSubgrsEdges_gotB = find_int_edges_lbl3(dgtrs_extEdgesB,
#                                                                      dgtrs_extEdges_dictsB, bitVecsB, edgeKey2lbl,
#                                                                      dmrs_node2bitVec)
        shrg_keyA = (dgtr_derivRules, interSubgrsEdges_gotA)
#         print (shrg_keyA)
        if shrg_keyA in SHRG:
#             print (mode, shrg_keyA)
            extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                                     dgtrs_extEdges_dictsA, edgeKey2lbl)
            shrgKey2psvItem[(shrg_keyA, dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs, scopes,
                                          extEdges_newA, extEdges_dicts_newA, dgtrs_extEdges,
                                          extEdgeKeyExts_listA, interSubgrsEdges_keysA)
        if not directed:
            interSubgrsEdges_gotB = (interSubgrsEdges_gotA[2], interSubgrsEdges_gotA[3], interSubgrsEdges_gotA[0],
                                     interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[5], interSubgrsEdges_gotA[4])
            interSubgrsEdges_keysB = (interSubgrsEdges_keysA[2], interSubgrsEdges_keysA[3],
                                      interSubgrsEdges_keysA[0], interSubgrsEdges_keysA[1],
                                      interSubgrsEdges_keysA[5], interSubgrsEdges_keysA[4])
            shrg_keyB = ((dgtr_derivRules[0], None, dgtr_derivRules[1]), interSubgrsEdges_gotB)
            if shrg_keyB in SHRG:
    #             print (mode, shrg_keyB)
                dgtr_derivRules_exactB = (dgtr_derivRules_exact[0], dgtr_derivRules_exact[2], dgtr_derivRules_exact[1])
                dgtrs_extEdgesB = (dgtrs_extEdges[0], dgtrs_extEdges[2], dgtrs_extEdges[1])
                dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[2], *dgtrs_extEdges_dicts[1])
                bitVecsB = (bitVecs[0], bitVecs[2], bitVecs[1])
                dgtrs_trgsB = (dgtrs_trgs[0], dgtrs_trgs[2], dgtrs_trgs[1])
                scopesB = (scopes[0], scopes[2], scopes[1])
                extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                        dgtrs_extEdges_dictsB, edgeKey2lbl)
                shrgKey2psvItem[(shrg_keyB, dgtr_derivRules_exactB)] = (bitVecsB, dgtrs_trgsB, scopesB,
                                              extEdges_newB, extEdges_dicts_newB,
                                             dgtrs_extEdgesB, extEdgeKeyExts_listB, interSubgrsEdges_keysB)
        return shrgKey2psvItem
    
    elif mode == 'BP': 
        dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[2])
        interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl3(dgtrs_extEdges,
                                                             dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                             dmrs_node2bitVec, node2pos
, node_typed)
        
        shrg_keyA = (dgtr_derivRules, interSubgrsEdges_gotA)
#         print (shrg_keyA)
#         print ()
        if shrg_keyA in SHRG:
#             print ("123312")
            extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                             dgtrs_extEdges_dictsA, edgeKey2lbl)
            shrgKey2psvItem[(shrg_keyA, dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs, scopes, extEdges_newA, extEdges_dicts_newA, 
                                          dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
        if not directed:
#         print (shrg_keyA)
            interSubgrsEdges_gotB = (interSubgrsEdges_gotA[2], interSubgrsEdges_gotA[3], interSubgrsEdges_gotA[0],
                                 interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[5], interSubgrsEdges_gotA[4])
            interSubgrsEdges_keysB = (interSubgrsEdges_keysA[2], interSubgrsEdges_keysA[3],
                                      interSubgrsEdges_keysA[0], interSubgrsEdges_keysA[1],
                                      interSubgrsEdges_keysA[5], interSubgrsEdges_keysA[4])
        
            shrg_keyB = ((dgtr_derivRules[0], dgtr_derivRules[2], dgtr_derivRules[1]), interSubgrsEdges_gotB)
    #         print (shrg_keyB)

            if shrg_keyB in SHRG:
    #             print (mode, shrg_keyB)
                dgtr_derivRules_exactB = (dgtr_derivRules_exact[0], dgtr_derivRules_exact[2], dgtr_derivRules_exact[1])
                dgtrs_extEdgesB = (dgtrs_extEdges[0], dgtrs_extEdges[2], dgtrs_extEdges[1])
                dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[2], *dgtrs_extEdges_dicts[1])
                bitVecsB = (bitVecs[0], bitVecs[2], bitVecs[1])
                dgtrs_trgsB = (dgtrs_trgs[0], dgtrs_trgs[2], dgtrs_trgs[1])
                scopesB = (scopes[0], scopes[2], scopes[1])
                extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                 dgtrs_extEdges_dictsB, edgeKey2lbl)
                shrgKey2psvItem[(shrg_keyB, dgtr_derivRules_exactB)] = (bitVecsB, dgtrs_trgsB, scopesB, extEdges_newB,
                                              extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB,
                                             interSubgrsEdges_keysB)
        return shrgKey2psvItem    
                
    elif mode == 'SubgrTag':
        # update new ext edge for the wsubgraph
        out_edges_list = [list(dmrs_nxDG_ext.out_edges(dgtr_node, keys = True, data = 'label')) for dgtr_node in dgtrs_nodes]
        in_edges_list = [list(dmrs_nxDG_ext.in_edges(dgtr_node, keys = True, data = 'label')) for dgtr_node in dgtrs_nodes]
#         print (list(zip(out_edges_list, in_edges_list)))
        dgtrs_extEdges_out = [{(s,t,k): ("s", get_src_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in out_edges} for out_edges in out_edges_list]
        dgtrs_extEdges_in = [{(s,t,k): ("t", get_targ_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in in_edges} for in_edges in in_edges_list]
        dgtrs_extEdges_dicts = []
        for idx, item in enumerate(dgtrs_extEdges_out):
            dgtrs_extEdges_dicts.append(item)
            dgtrs_extEdges_dicts.append(dgtrs_extEdges_in[idx])
#         print (dgtrs_extEdges_dicts)
    
        dgtrs_extEdges = tuple(tuple(sorted([((s,t,k), ("s", get_src_ext(l, node_typed))) if (s,t,k,l) in out_edges
                                                   else ((s,t,k),("t", get_targ_ext(l, node_typed)))
                                               for s,t,k,l in out_edges + in_edges], key = lambda x: x[1][1])
                                     )
                                for out_edges, in_edges in list(zip(out_edges_list, in_edges_list)))
        new_extEdge, new_extEdges_dicts, extEdgeKeyExts_list\
            = find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl)
#         print (new_extEdges_dicts)
        return new_extEdge, new_extEdges_dicts
    
    elif mode == 'Tag':
        out_edges = list(dmrs_nxDG_ext.out_edges(dgtrs_nodes[0], keys = True, data = 'label'))
        in_edges = list(dmrs_nxDG_ext.in_edges(dgtrs_nodes[0], keys = True, data = 'label'))
        dgtrs_extEdges_out = {(s,t,k): ("s", get_src_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in out_edges}
#         print (in_edges)
        dgtrs_extEdges_in = {(s,t,k): ("t", get_targ_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in in_edges}
        dgtrs_extEdges_dicts = []
        dgtrs_extEdges_dicts.append(dgtrs_extEdges_out)
        dgtrs_extEdges_dicts.append(dgtrs_extEdges_in)
#         print (dgtrs_extEdges_dicts)
        dgtrs_extEdges = (tuple(sorted([((s,t,k), ("s", get_src_ext(l, node_typed))) if (s,t,k,l) in out_edges
                                             else ((s,t,k),("t", get_targ_ext(l, node_typed)))
                                         for s,t,k,l in out_edges + in_edges], key = lambda x: x[1][1])),)
        new_extEdge, new_extEdges_dicts, extEdgeKeyExts_list =\
            find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl)
#         print (dgtrs_nodes, new_extEdge)
        return new_extEdge, new_extEdges_dicts

    
def _U(dmrs_nxDG_ext, dgtr_derivRules_prob, dgtr_derivRules_exact, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts, dgtrs_trgs, scopes, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items, mode, node2pos = None, node_typed = False, p = False, directed = False, return_exts = False):
    shrgKey2psvItem = defaultdict()
    if dgtr_derivRules in SHRG:
        shrgKey2psvItem[dgtr_derivRules] = dgtrs_extEdges[0], dgtrs_extEdges_dicts[0]
    return shrgKey2psvItem

def _B(dmrs_nxDG_ext, dgtr_derivRules_prob, dgtr_derivRules_exact, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts, dgtrs_trgs, scopes, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items, mode, node2pos = None, node_typed = False, p = False, directed = False, return_exts = False):
    shrgKey2psvItem = defaultdict()
    dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1])
    interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl2(dgtrs_extEdges,
                                                       dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                         dmrs_node2bitVec, node2pos, node_typed)
    shrg_keyA = (dgtr_derivRules, interSubgrsEdges_gotA)
    if shrg_keyA in SHRG:
        extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                                 dgtrs_extEdges_dictsA, edgeKey2lbl)
        shrgKey2psvItem[(shrg_keyA, dgtr_derivRules_exact, dgtr_derivRules_prob)]\
            = (bitVecs, dgtrs_trgs, scopes, extEdges_newA, extEdges_dicts_newA,
                                      dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
    if not directed:
        interSubgrsEdges_gotB = (interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[0])
        interSubgrsEdges_keysB = (interSubgrsEdges_keysA[1], interSubgrsEdges_keysA[0])
        shrg_keyB = ((dgtr_derivRules[1], dgtr_derivRules[0]), interSubgrsEdges_gotB)
        if shrg_keyB in SHRG:
            # print (mode, shrg_keyB)
            dgtr_derivRules_probB = (dgtr_derivRules_prob[1], dgtr_derivRules_prob[0])
            dgtr_derivRules_exactB = (dgtr_derivRules_exact[1], dgtr_derivRules_exact[0])
            dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[0])
            dgtrs_extEdgesB = (dgtrs_extEdges[1], dgtrs_extEdges[0])
            bitVecsB = (bitVecs[1], bitVecs[0])
            dgtrs_trgsB = (dgtrs_trgs[1], dgtrs_trgs[0])
            scopesB = (scopes[1], scopes[0])
            extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                     dgtrs_extEdges_dictsB, edgeKey2lbl) 
            shrgKey2psvItem[(shrg_keyB, dgtr_derivRules_exactB, dgtr_derivRules_probB)]\
                = (bitVecsB, dgtrs_trgsB, scopesB, extEdges_newB,
                                          extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB
                                         , interSubgrsEdges_keysB)
    return shrgKey2psvItem

def _BpredSemEmt(dmrs_nxDG_ext, dgtr_derivRules_prob, dgtr_derivRules_exact, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts, dgtrs_trgs, scopes, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items, mode, node2pos = None, node_typed = False, p = False, directed = False, return_exts = False):
    shrgKey2psvItem = defaultdict()
    dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1])
    interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl2(dgtrs_extEdges,
                                                       dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                         dmrs_node2bitVec, node2pos, node_typed)
    shrg_keyA = interSubgrsEdges_gotA
    if shrg_keyA in SHRG[dgtr_derivRules]:
        extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                                 dgtrs_extEdges_dictsA, edgeKey2lbl)
        shrgKey2psvItem[((dgtr_derivRules, shrg_keyA), dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs,
                                                                                  scopes, extEdges_newA,
                                                         extEdges_dicts_newA,
                                      dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
    if not directed:
        interSubgrsEdges_gotB = (interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[0])
        interSubgrsEdges_keysB = (interSubgrsEdges_keysA[1], interSubgrsEdges_keysA[0])
        shrg_keyB = interSubgrsEdges_gotB
        if shrg_keyB in SHRG[(dgtr_derivRules[1], dgtr_derivRules[0])]:
            dgtr_derivRules_exactB = (dgtr_derivRules_exact[1], dgtr_derivRules_exact[0])
            dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[0])
            dgtrs_extEdgesB = (dgtrs_extEdges[1], dgtrs_extEdges[0])
            bitVecsB = (bitVecs[1], bitVecs[0])
            dgtrs_trgsB = (dgtrs_trgs[1], dgtrs_trgs[0])
            scopesB = (scopes[1], scopes[0])
            extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                     dgtrs_extEdges_dictsB, edgeKey2lbl) 
            shrgKey2psvItem[(((dgtr_derivRules[1], dgtr_derivRules[0]), shrg_keyB),
                             dgtr_derivRules_exactB)] = (bitVecsB, dgtrs_trgsB,
                                                                                      scopesB, extEdges_newB,
                                          extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB
                                         , interSubgrsEdges_keysB)
    return shrgKey2psvItem


def _BPpredSemEmt(dmrs_nxDG_ext, dgtr_derivRules_prob, dgtr_derivRules_exact, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts, dgtrs_trgs, scopes, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items, mode, node2pos = None, node_typed = False, p = False, directed = False, return_exts = False):
    shrgKey2psvItem = defaultdict()
    dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[2])
    interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl3(dgtrs_extEdges,
                                                         dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                         dmrs_node2bitVec, node2pos, node_typed)
    
    shrg_keyA = interSubgrsEdges_gotA
    if shrg_keyA in SHRG[dgtr_derivRules]:
        extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                         dgtrs_extEdges_dictsA, edgeKey2lbl)
        shrgKey2psvItem[((dgtr_derivRules, shrg_keyA), dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs, scopes, extEdges_newA,
                                                         extEdges_dicts_newA, 
                                      dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
    if not directed:
        interSubgrsEdges_gotB = (interSubgrsEdges_gotA[2], interSubgrsEdges_gotA[3], interSubgrsEdges_gotA[0],
                             interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[5], interSubgrsEdges_gotA[4])
        interSubgrsEdges_keysB = (interSubgrsEdges_keysA[2], interSubgrsEdges_keysA[3],
                                  interSubgrsEdges_keysA[0], interSubgrsEdges_keysA[1],
                                  interSubgrsEdges_keysA[5], interSubgrsEdges_keysA[4])
    
        shrg_keyB = interSubgrsEdges_gotB

        if shrg_keyB in SHRG[(dgtr_derivRules[0], dgtr_derivRules[2], dgtr_derivRules[1])]:
            dgtr_derivRules_exactB = (dgtr_derivRules_exact[0], dgtr_derivRules_exact[2], dgtr_derivRules_exact[1])
            dgtrs_extEdgesB = (dgtrs_extEdges[0], dgtrs_extEdges[2], dgtrs_extEdges[1])
            dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[2], *dgtrs_extEdges_dicts[1])
            bitVecsB = (bitVecs[0], bitVecs[2], bitVecs[1])
            dgtrs_trgsB = (dgtrs_trgs[0], dgtrs_trgs[2], dgtrs_trgs[1])
            scopesB = (scopes[0], scopes[2], scopes[1])
            extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                             dgtrs_extEdges_dictsB, edgeKey2lbl)
            shrgKey2psvItem[(((dgtr_derivRules[0], dgtr_derivRules[2], dgtr_derivRules[1]), shrg_keyB),
                           dgtr_derivRules_exactB)]\
                = (bitVecsB, dgtrs_trgsB, scopesB, extEdges_newB,
                                          extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB,
                                         interSubgrsEdges_keysB)
    return shrgKey2psvItem    
            

def _BPsemEmt(dmrs_nxDG_ext, dgtr_derivRules_prob, dgtr_derivRules_exact, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts, dgtrs_trgs, scopes, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items, mode, node2pos = None, node_typed = False, p = False, directed = False, return_exts = False):
    shrgKey2psvItem = defaultdict()
    extEdges_newA = None
    dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[2])
    interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl3(dgtrs_extEdges,
                                                        dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                         dmrs_node2bitVec, node2pos,
node_typed)
    shrg_keyA = (dgtr_derivRules, interSubgrsEdges_gotA)
    if shrg_keyA in SHRG:
        extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                                 dgtrs_extEdges_dictsA, edgeKey2lbl)
        shrgKey2psvItem[(shrg_keyA, dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs, scopes,
                                      extEdges_newA, extEdges_dicts_newA, dgtrs_extEdges,
                                      extEdgeKeyExts_listA, interSubgrsEdges_keysA)
    if not directed:
        interSubgrsEdges_gotB = (interSubgrsEdges_gotA[2], interSubgrsEdges_gotA[3], interSubgrsEdges_gotA[0],
                                 interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[5], interSubgrsEdges_gotA[4])
        interSubgrsEdges_keysB = (interSubgrsEdges_keysA[2], interSubgrsEdges_keysA[3],
                                  interSubgrsEdges_keysA[0], interSubgrsEdges_keysA[1],
                                  interSubgrsEdges_keysA[5], interSubgrsEdges_keysA[4])
        shrg_keyB = ((dgtr_derivRules[0], None, dgtr_derivRules[1]), interSubgrsEdges_gotB)
        if shrg_keyB in SHRG:
            dgtr_derivRules_exactB = (dgtr_derivRules_exact[0], dgtr_derivRules_exact[2], dgtr_derivRules_exact[1])
            dgtrs_extEdgesB = (dgtrs_extEdges[0], dgtrs_extEdges[2], dgtrs_extEdges[1])
            dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[2], *dgtrs_extEdges_dicts[1])
            bitVecsB = (bitVecs[0], bitVecs[2], bitVecs[1])
            dgtrs_trgsB = (dgtrs_trgs[0], dgtrs_trgs[2], dgtrs_trgs[1])
            scopesB = (scopes[0], scopes[2], scopes[1])
            extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                    dgtrs_extEdges_dictsB, edgeKey2lbl)
            shrgKey2psvItem[(shrg_keyB, dgtr_derivRules_exactB)] = (bitVecsB, dgtrs_trgsB, scopesB,
                                          extEdges_newB, extEdges_dicts_newB,
                                         dgtrs_extEdgesB, extEdgeKeyExts_listB, interSubgrsEdges_keysB)
    return shrgKey2psvItem

def _BP(dmrs_nxDG_ext, dgtr_derivRules_prob, dgtr_derivRules_exact, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts, dgtrs_trgs, scopes, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items, mode, node2pos = None, node_typed = False, p = False, directed = False, return_exts = False): 
    shrgKey2psvItem = defaultdict()
    dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[2])
    interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl3(dgtrs_extEdges,
                                                         dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                         dmrs_node2bitVec, node2pos, node_typed)
    
    shrg_keyA = (dgtr_derivRules, interSubgrsEdges_gotA)
    if shrg_keyA in SHRG:
        extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                         dgtrs_extEdges_dictsA, edgeKey2lbl)
        shrgKey2psvItem[(shrg_keyA, dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs, scopes, extEdges_newA, extEdges_dicts_newA, 
                                      dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
    if not directed:
        interSubgrsEdges_gotB = (interSubgrsEdges_gotA[2], interSubgrsEdges_gotA[3], interSubgrsEdges_gotA[0],
                             interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[5], interSubgrsEdges_gotA[4])
        interSubgrsEdges_keysB = (interSubgrsEdges_keysA[2], interSubgrsEdges_keysA[3],
                                  interSubgrsEdges_keysA[0], interSubgrsEdges_keysA[1],
                                  interSubgrsEdges_keysA[5], interSubgrsEdges_keysA[4])
    
        shrg_keyB = ((dgtr_derivRules[0], dgtr_derivRules[2], dgtr_derivRules[1]), interSubgrsEdges_gotB)

        if shrg_keyB in SHRG:
            dgtr_derivRules_exactB = (dgtr_derivRules_exact[0], dgtr_derivRules_exact[2], dgtr_derivRules_exact[1])
            dgtrs_extEdgesB = (dgtrs_extEdges[0], dgtrs_extEdges[2], dgtrs_extEdges[1])
            dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[2], *dgtrs_extEdges_dicts[1])
            bitVecsB = (bitVecs[0], bitVecs[2], bitVecs[1])
            dgtrs_trgsB = (dgtrs_trgs[0], dgtrs_trgs[2], dgtrs_trgs[1])
            scopesB = (scopes[0], scopes[2], scopes[1])
            extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                             dgtrs_extEdges_dictsB, edgeKey2lbl)
            shrgKey2psvItem[(shrg_keyB, dgtr_derivRules_exactB)] = (bitVecsB, dgtrs_trgsB, scopesB, extEdges_newB,
                                          extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB,
                                         interSubgrsEdges_keysB)
    return shrgKey2psvItem    
            
def _SubgrTag(dmrs_nxDG_ext, dgtr_derivRules_prob, dgtr_derivRules_exact, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts, dgtrs_trgs, scopes, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items, mode, node2pos = None, node_typed = False, p = False, directed = False, return_exts = False):
    # update new ext edge for the wsubgraph
    out_edges_list = [list(dmrs_nxDG_ext.out_edges(dgtr_node, keys = True, data = 'label')) for dgtr_node in dgtrs_nodes]
    in_edges_list = [list(dmrs_nxDG_ext.in_edges(dgtr_node, keys = True, data = 'label')) for dgtr_node in dgtrs_nodes]
    dgtrs_extEdges_out = [{(s,t,k): ("s", get_src_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                            for s,t,k,l in out_edges} for out_edges in out_edges_list]
    dgtrs_extEdges_in = [{(s,t,k): ("t", get_targ_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                            for s,t,k,l in in_edges} for in_edges in in_edges_list]
    dgtrs_extEdges_dicts = []
    for idx, item in enumerate(dgtrs_extEdges_out):
        dgtrs_extEdges_dicts.append(item)
        dgtrs_extEdges_dicts.append(dgtrs_extEdges_in[idx])

    dgtrs_extEdges = tuple(tuple(sorted([((s,t,k), ("s", get_src_ext(l, node_typed))) if (s,t,k,l) in out_edges
                                               else ((s,t,k),("t", get_targ_ext(l, node_typed)))
                                           for s,t,k,l in out_edges + in_edges], key = lambda x: x[1][1])
                                 )
                            for out_edges, in_edges in list(zip(out_edges_list, in_edges_list)))
    new_extEdge, new_extEdges_dicts, extEdgeKeyExts_list\
        = find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl)
    return new_extEdge, new_extEdges_dicts

def _Tag(dmrs_nxDG_ext, dgtr_derivRules_prob, dgtr_derivRules_exact, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts, dgtrs_trgs, scopes, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items, mode, node2pos = None, node_typed = False, p = False, directed = False, return_exts = False):
    out_edges = list(dmrs_nxDG_ext.out_edges(dgtrs_nodes[0], keys = True, data = 'label'))
    in_edges = list(dmrs_nxDG_ext.in_edges(dgtrs_nodes[0], keys = True, data = 'label'))
    dgtrs_extEdges_out = {(s,t,k): ("s", get_src_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                            for s,t,k,l in out_edges}
    dgtrs_extEdges_in = {(s,t,k): ("t", get_targ_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                            for s,t,k,l in in_edges}
    dgtrs_extEdges_dicts = []
    dgtrs_extEdges_dicts.append(dgtrs_extEdges_out)
    dgtrs_extEdges_dicts.append(dgtrs_extEdges_in)
    dgtrs_extEdges = (tuple(sorted([((s,t,k), ("s", get_src_ext(l, node_typed))) if (s,t,k,l) in out_edges
                                         else ((s,t,k),("t", get_targ_ext(l, node_typed)))
                                     for s,t,k,l in out_edges + in_edges], key = lambda x: x[1][1])),)
    new_extEdge, new_extEdges_dicts, extEdgeKeyExts_list =\
        find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl)
    return new_extEdge, new_extEdges_dicts

semCompMode2func = {
            "U": _U,
             "B": _B,
             "B-predSemEmt": _BpredSemEmt,
             "BP-predSemEmt": _BPpredSemEmt,
             "BPsemEmt": _BPsemEmt,
             "BP": _BP,
             "SubgrTag": _SubgrTag,
             "Tag": _Tag
             }

# old, correct
def matched_semComp_lazy(dmrs_nxDG_ext, dgtr_derivRules_prob, dgtr_derivRules_exact, dgtr_derivRules, dgtrs_nodes, dgtrs_extEdges, dgtrs_extEdges_dicts,
                         dgtrs_trgs, scopes, bitVecs, SHRG, edgeKey2lbl, dmrs_node2bitVec, no_items,
                         mode, node2pos = None, node_typed = False, p = False, directed = False, return_exts = False):
    
    shrgKey2psvItem = defaultdict()
    if mode == 'U':
#         print (dgtr_derivRules)
#         print (SHRG.keys())
        if dgtr_derivRules in SHRG:
            # print ("v;", SHRG[dgtr_derivRules])
            shrgKey2psvItem[dgtr_derivRules] = dgtrs_extEdges[0], dgtrs_extEdges_dicts[0]
        return shrgKey2psvItem
    
    elif mode == 'B':
        dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1])
        interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl2(dgtrs_extEdges,
                                                           dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                             dmrs_node2bitVec, node2pos
, node_typed)
        shrg_keyA = (dgtr_derivRules, interSubgrsEdges_gotA)
#         if p:
        # print (shrg_keyA)
        # for k in SHRG:
        #     print (k)
        #     input()
#             print ()
#             print (SHRG)
#             print ()
#             print ()
        if shrg_keyA in SHRG:
            # print (mode, shrg_keyA)
            extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                                     dgtrs_extEdges_dictsA, edgeKey2lbl)
            shrgKey2psvItem[(shrg_keyA, dgtr_derivRules_exact, dgtr_derivRules_prob)]\
                = (bitVecs, dgtrs_trgs, scopes, extEdges_newA, extEdges_dicts_newA,
                                          dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
        if not directed:
            interSubgrsEdges_gotB = (interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[0])
            interSubgrsEdges_keysB = (interSubgrsEdges_keysA[1], interSubgrsEdges_keysA[0])
            shrg_keyB = ((dgtr_derivRules[1], dgtr_derivRules[0]), interSubgrsEdges_gotB)
            if shrg_keyB in SHRG:
                # print (mode, shrg_keyB)
                dgtr_derivRules_probB = (dgtr_derivRules_prob[1], dgtr_derivRules_prob[0])
                dgtr_derivRules_exactB = (dgtr_derivRules_exact[1], dgtr_derivRules_exact[0])
                dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[0])
                dgtrs_extEdgesB = (dgtrs_extEdges[1], dgtrs_extEdges[0])
                bitVecsB = (bitVecs[1], bitVecs[0])
                dgtrs_trgsB = (dgtrs_trgs[1], dgtrs_trgs[0])
                scopesB = (scopes[1], scopes[0])
                extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                         dgtrs_extEdges_dictsB, edgeKey2lbl) 
                shrgKey2psvItem[(shrg_keyB, dgtr_derivRules_exactB, dgtr_derivRules_probB)]\
                    = (bitVecsB, dgtrs_trgsB, scopesB, extEdges_newB,
                                              extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB
                                             , interSubgrsEdges_keysB)
        return shrgKey2psvItem
    
    elif mode == 'B-predSemEmt':
        dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1])
        interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl2(dgtrs_extEdges,
                                                           dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                             dmrs_node2bitVec, node2pos
, node_typed)
        shrg_keyA = interSubgrsEdges_gotA
#         if p:
#             print (shrg_keyA)
#             print ()
#             print (SHRG)
#             print ()
#             print ()
        if shrg_keyA in SHRG[dgtr_derivRules]:
#             print (mode, shrg_keyA)
            extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                                     dgtrs_extEdges_dictsA, edgeKey2lbl)
            shrgKey2psvItem[((dgtr_derivRules, shrg_keyA), dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs,
                                                                                      scopes, extEdges_newA,
                                                             extEdges_dicts_newA,
                                          dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
        if not directed:
            interSubgrsEdges_gotB = (interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[0])
            interSubgrsEdges_keysB = (interSubgrsEdges_keysA[1], interSubgrsEdges_keysA[0])
            shrg_keyB = interSubgrsEdges_gotB
            if shrg_keyB in SHRG[(dgtr_derivRules[1], dgtr_derivRules[0])]:
#             print (mode, shrg_keyB)
                dgtr_derivRules_exactB = (dgtr_derivRules_exact[1], dgtr_derivRules_exact[0])
                dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[0])
                dgtrs_extEdgesB = (dgtrs_extEdges[1], dgtrs_extEdges[0])
                bitVecsB = (bitVecs[1], bitVecs[0])
                dgtrs_trgsB = (dgtrs_trgs[1], dgtrs_trgs[0])
                scopesB = (scopes[1], scopes[0])
                extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                         dgtrs_extEdges_dictsB, edgeKey2lbl) 
                shrgKey2psvItem[(((dgtr_derivRules[1], dgtr_derivRules[0]), shrg_keyB),
                                 dgtr_derivRules_exactB)] = (bitVecsB, dgtrs_trgsB,
                                                                                          scopesB, extEdges_newB,
                                              extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB
                                             , interSubgrsEdges_keysB)
        return shrgKey2psvItem
    
    
    elif mode == 'BP-predSemEmt':
        dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[2])
        interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl3(dgtrs_extEdges,
                                                             dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                             dmrs_node2bitVec, node2pos
, node_typed)
        
        shrg_keyA = interSubgrsEdges_gotA
#         print ()
        if shrg_keyA in SHRG[dgtr_derivRules]:
#             print ("123312")
            extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                             dgtrs_extEdges_dictsA, edgeKey2lbl)
            shrgKey2psvItem[((dgtr_derivRules, shrg_keyA), dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs, scopes, extEdges_newA,
                                                             extEdges_dicts_newA, 
                                          dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
        if not directed:
#         print (shrg_keyA)
            interSubgrsEdges_gotB = (interSubgrsEdges_gotA[2], interSubgrsEdges_gotA[3], interSubgrsEdges_gotA[0],
                                 interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[5], interSubgrsEdges_gotA[4])
            interSubgrsEdges_keysB = (interSubgrsEdges_keysA[2], interSubgrsEdges_keysA[3],
                                      interSubgrsEdges_keysA[0], interSubgrsEdges_keysA[1],
                                      interSubgrsEdges_keysA[5], interSubgrsEdges_keysA[4])
        
            shrg_keyB = interSubgrsEdges_gotB
    #         print (shrg_keyB)

            if shrg_keyB in SHRG[(dgtr_derivRules[0], dgtr_derivRules[2], dgtr_derivRules[1])]:
    #             print (mode, shrg_keyB)
                dgtr_derivRules_exactB = (dgtr_derivRules_exact[0], dgtr_derivRules_exact[2], dgtr_derivRules_exact[1])
                dgtrs_extEdgesB = (dgtrs_extEdges[0], dgtrs_extEdges[2], dgtrs_extEdges[1])
                dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[2], *dgtrs_extEdges_dicts[1])
                bitVecsB = (bitVecs[0], bitVecs[2], bitVecs[1])
                dgtrs_trgsB = (dgtrs_trgs[0], dgtrs_trgs[2], dgtrs_trgs[1])
                scopesB = (scopes[0], scopes[2], scopes[1])
                extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                 dgtrs_extEdges_dictsB, edgeKey2lbl)
                shrgKey2psvItem[(((dgtr_derivRules[0], dgtr_derivRules[2], dgtr_derivRules[1]), shrg_keyB),
                               dgtr_derivRules_exactB)]\
                    = (bitVecsB, dgtrs_trgsB, scopesB, extEdges_newB,
                                              extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB,
                                             interSubgrsEdges_keysB)
        return shrgKey2psvItem    
                
    
    elif mode == 'BPsemEmt':
        extEdges_newA = None
        dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[2])
        interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl3(dgtrs_extEdges,
                                                            dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                             dmrs_node2bitVec, node2pos
, node_typed)
#         _, _, interSubgrsEdges_gotB = find_int_edges_lbl3(dgtrs_extEdgesB,
#                                                                      dgtrs_extEdges_dictsB, bitVecsB, edgeKey2lbl,
#                                                                      dmrs_node2bitVec)
        shrg_keyA = (dgtr_derivRules, interSubgrsEdges_gotA)
#         print (shrg_keyA)
        if shrg_keyA in SHRG:
#             print (mode, shrg_keyA)
            extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                                     dgtrs_extEdges_dictsA, edgeKey2lbl)
            shrgKey2psvItem[(shrg_keyA, dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs, scopes,
                                          extEdges_newA, extEdges_dicts_newA, dgtrs_extEdges,
                                          extEdgeKeyExts_listA, interSubgrsEdges_keysA)
        if not directed:
            interSubgrsEdges_gotB = (interSubgrsEdges_gotA[2], interSubgrsEdges_gotA[3], interSubgrsEdges_gotA[0],
                                     interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[5], interSubgrsEdges_gotA[4])
            interSubgrsEdges_keysB = (interSubgrsEdges_keysA[2], interSubgrsEdges_keysA[3],
                                      interSubgrsEdges_keysA[0], interSubgrsEdges_keysA[1],
                                      interSubgrsEdges_keysA[5], interSubgrsEdges_keysA[4])
            shrg_keyB = ((dgtr_derivRules[0], None, dgtr_derivRules[1]), interSubgrsEdges_gotB)
            if shrg_keyB in SHRG:
    #             print (mode, shrg_keyB)
                dgtr_derivRules_exactB = (dgtr_derivRules_exact[0], dgtr_derivRules_exact[2], dgtr_derivRules_exact[1])
                dgtrs_extEdgesB = (dgtrs_extEdges[0], dgtrs_extEdges[2], dgtrs_extEdges[1])
                dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[2], *dgtrs_extEdges_dicts[1])
                bitVecsB = (bitVecs[0], bitVecs[2], bitVecs[1])
                dgtrs_trgsB = (dgtrs_trgs[0], dgtrs_trgs[2], dgtrs_trgs[1])
                scopesB = (scopes[0], scopes[2], scopes[1])
                extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                        dgtrs_extEdges_dictsB, edgeKey2lbl)
                shrgKey2psvItem[(shrg_keyB, dgtr_derivRules_exactB)] = (bitVecsB, dgtrs_trgsB, scopesB,
                                              extEdges_newB, extEdges_dicts_newB,
                                             dgtrs_extEdgesB, extEdgeKeyExts_listB, interSubgrsEdges_keysB)
        return shrgKey2psvItem
    
    elif mode == 'BP': 
        dgtrs_extEdges_dictsA = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[1], *dgtrs_extEdges_dicts[2])
        interSubgrsEdges_gotA, interSubgrsEdges_keysA = find_int_edges_lbl3(dgtrs_extEdges,
                                                             dgtrs_extEdges_dictsA, bitVecs, edgeKey2lbl,
                                                             dmrs_node2bitVec, node2pos
, node_typed)
        
        shrg_keyA = (dgtr_derivRules, interSubgrsEdges_gotA)
#         print (shrg_keyA)
#         print ()
        if shrg_keyA in SHRG:
#             print ("123312")
            extEdges_newA, extEdges_dicts_newA, extEdgeKeyExts_listA = find_ext_edges_lbl(dgtrs_extEdges,
                                                             dgtrs_extEdges_dictsA, edgeKey2lbl)
            shrgKey2psvItem[(shrg_keyA, dgtr_derivRules_exact)] = (bitVecs, dgtrs_trgs, scopes, extEdges_newA, extEdges_dicts_newA, 
                                          dgtrs_extEdges, extEdgeKeyExts_listA, interSubgrsEdges_keysA)
        if not directed:
#         print (shrg_keyA)
            interSubgrsEdges_gotB = (interSubgrsEdges_gotA[2], interSubgrsEdges_gotA[3], interSubgrsEdges_gotA[0],
                                 interSubgrsEdges_gotA[1], interSubgrsEdges_gotA[5], interSubgrsEdges_gotA[4])
            interSubgrsEdges_keysB = (interSubgrsEdges_keysA[2], interSubgrsEdges_keysA[3],
                                      interSubgrsEdges_keysA[0], interSubgrsEdges_keysA[1],
                                      interSubgrsEdges_keysA[5], interSubgrsEdges_keysA[4])
        
            shrg_keyB = ((dgtr_derivRules[0], dgtr_derivRules[2], dgtr_derivRules[1]), interSubgrsEdges_gotB)
    #         print (shrg_keyB)

            if shrg_keyB in SHRG:
    #             print (mode, shrg_keyB)
                dgtr_derivRules_exactB = (dgtr_derivRules_exact[0], dgtr_derivRules_exact[2], dgtr_derivRules_exact[1])
                dgtrs_extEdgesB = (dgtrs_extEdges[0], dgtrs_extEdges[2], dgtrs_extEdges[1])
                dgtrs_extEdges_dictsB = (*dgtrs_extEdges_dicts[0], *dgtrs_extEdges_dicts[2], *dgtrs_extEdges_dicts[1])
                bitVecsB = (bitVecs[0], bitVecs[2], bitVecs[1])
                dgtrs_trgsB = (dgtrs_trgs[0], dgtrs_trgs[2], dgtrs_trgs[1])
                scopesB = (scopes[0], scopes[2], scopes[1])
                extEdges_newB, extEdges_dicts_newB, extEdgeKeyExts_listB = find_ext_edges_lbl(dgtrs_extEdgesB,
                                                                 dgtrs_extEdges_dictsB, edgeKey2lbl)
                shrgKey2psvItem[(shrg_keyB, dgtr_derivRules_exactB)] = (bitVecsB, dgtrs_trgsB, scopesB, extEdges_newB,
                                              extEdges_dicts_newB, dgtrs_extEdgesB, extEdgeKeyExts_listB,
                                             interSubgrsEdges_keysB)
        return shrgKey2psvItem    
                
    elif mode == 'SubgrTag':
        # update new ext edge for the wsubgraph
        out_edges_list = [list(dmrs_nxDG_ext.out_edges(dgtr_node, keys = True, data = 'label')) for dgtr_node in dgtrs_nodes]
        in_edges_list = [list(dmrs_nxDG_ext.in_edges(dgtr_node, keys = True, data = 'label')) for dgtr_node in dgtrs_nodes]
#         print (list(zip(out_edges_list, in_edges_list)))
        dgtrs_extEdges_out = [{(s,t,k): ("s", get_src_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in out_edges} for out_edges in out_edges_list]
        dgtrs_extEdges_in = [{(s,t,k): ("t", get_targ_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in in_edges} for in_edges in in_edges_list]
        dgtrs_extEdges_dicts = []
        for idx, item in enumerate(dgtrs_extEdges_out):
            dgtrs_extEdges_dicts.append(item)
            dgtrs_extEdges_dicts.append(dgtrs_extEdges_in[idx])
#         print (dgtrs_extEdges_dicts)
    
        dgtrs_extEdges = tuple(tuple(sorted([((s,t,k), ("s", get_src_ext(l, node_typed))) if (s,t,k,l) in out_edges
                                                   else ((s,t,k),("t", get_targ_ext(l, node_typed)))
                                               for s,t,k,l in out_edges + in_edges], key = lambda x: x[1][1])
                                     )
                                for out_edges, in_edges in list(zip(out_edges_list, in_edges_list)))
        new_extEdge, new_extEdges_dicts, extEdgeKeyExts_list\
            = find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl)
#         print (new_extEdges_dicts)
        return new_extEdge, new_extEdges_dicts
    
    elif mode == 'Tag':
        out_edges = list(dmrs_nxDG_ext.out_edges(dgtrs_nodes[0], keys = True, data = 'label'))
        in_edges = list(dmrs_nxDG_ext.in_edges(dgtrs_nodes[0], keys = True, data = 'label'))
        dgtrs_extEdges_out = {(s,t,k): ("s", get_src_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in out_edges}
#         print (in_edges)
        dgtrs_extEdges_in = {(s,t,k): ("t", get_targ_ext(l, node_typed), edgeKey2lbl[(s,t,k)])
                                for s,t,k,l in in_edges}
        dgtrs_extEdges_dicts = []
        dgtrs_extEdges_dicts.append(dgtrs_extEdges_out)
        dgtrs_extEdges_dicts.append(dgtrs_extEdges_in)
#         print (dgtrs_extEdges_dicts)
        dgtrs_extEdges = (tuple(sorted([((s,t,k), ("s", get_src_ext(l, node_typed))) if (s,t,k,l) in out_edges
                                             else ((s,t,k),("t", get_targ_ext(l, node_typed)))
                                         for s,t,k,l in out_edges + in_edges], key = lambda x: x[1][1])),)
        new_extEdge, new_extEdges_dicts, extEdgeKeyExts_list =\
            find_ext_edges_lbl(dgtrs_extEdges, dgtrs_extEdges_dicts, edgeKey2lbl)
#         print (dgtrs_nodes, new_extEdge)
        return new_extEdge, new_extEdges_dicts
            
#     return shrgKey2psvItems
def index_aval_check(derivRule, extEdgeKeyExts_list, edgeKey2lbl, no_items, p = False):
    # index availability check
    index_aval = True
    derivRule_orig = derivRule.split("&")[0].split("^")[0]
#     print (derivRule_orig)
    if derivRule_orig in erg_rules and "_crd" in derivRule_orig or "mrk-" in derivRule_orig:
        pass
    elif derivRule_orig in half_erg_rules and "j-j_" in derivRule_orig\
        or "pp-pp" in derivRule_orig:
        pass
    elif erg_rule2hdness.get(derivRule_orig)[-2:] == "hd":
        if erg_rule2hdness.get(derivRule_orig) == "B-lhd":
#             if p:
#                 pprint (extEdgeKeyExts_list)
            for key, ext in extEdgeKeyExts_list[-1]:
                if ext[0] == 't' and "/EQ" in edgeKey2lbl[key]:# or ext[0] == 't' and "/NEQ" in edgeKey2lbl[key]: #xarg
#                     print (key, edgeKey2lbl[key])
                    index_aval = False
                    break
            # case of unary non-hd rule above current rule so don't check
#             if no_items == 3:
#                 for key, ext in extEdgeKeyExts_list[0]:
#                     if ext[0] == 't' and "/EQ" in edgeKey2lbl[key] or "/NEQ" in edgeKey2lbl[key]:
#                         index_aval = False
#                         break
        elif erg_rule2hdness.get(derivRule_orig) == "B-rhd":
            for key, ext in extEdgeKeyExts_list[-2]:
                if ext[0] == 't' and "/EQ" in edgeKey2lbl[key]:# or ext[0] == 't' and "/NEQ" in edgeKey2lbl[key]: #xarg
#                     print (key, edgeKey2lbl[key])
                    index_aval = False
                    break
#             if no_items == 3:
#                 for key, ext in extEdgeKeyExts_list[0]:
#                     if ext[0] == 't' and "/EQ" in edgeKey2lbl[key] or "/NEQ" in edgeKey2lbl[key]:
#                         index_aval = False
#                         break
        elif erg_rule2hdness.get(derivRule_orig) == "U-hd":
            pass
    
    
    return index_aval
        
def ltop_aval_check(derivRule, extEdgeKeyExts_list, interSubgrsEdges, edgeKey2lbl, scopes, dmrsNode2scope,
                     no_items, p = False):           
    # extract scopal order
    # ignore if multiple ltop possible/insufficient info about dgtrs' ltop
    ltop_aval = True
    outest_scope = None
    edges_set_idxs_list = ()
    multi_ltop = False
    if no_items == 3:
        edges_set_idxs_list = ((0, 2), (1, 4), (3, 5))
        # check incoming "/H or /HEQ" edges here (or use them to decide on new ltop)
        outest_scope_new = None
        for extEdgeKeyExts in extEdgeKeyExts_list:
            for key, ext in extEdgeKeyExts:
                if ext[0] == 't' and ("/H" in edgeKey2lbl[key]\
                                      or "/HEQ" in edgeKey2lbl[key] or "/EQ" in edgeKey2lbl[key]):
                    if outest_scope == None:
                        if outest_scope_new != None and outest_scope_new != dmrsNode2scope[key[1]]:
                            # multiple new ltop; treat as available
                            outest_scope_new = None
                            multi_ltop = True
                            break
                        else: outest_scope_new = dmrsNode2scope[key[1]]
                    elif outest_scope != None:
                        if dmrsNode2scope[key[1]] != outest_scope:
                            ltop_aval = False
    #                         print ("ltop not aval:", derivRule, key, edgeKey2lbl[key])
                            break
            if multi_ltop or not ltop_aval: break
        if outest_scope == None and outest_scope_new != None:
            outest_scope = outest_scope_new
        return ltop_aval, outest_scope
    elif no_items == 2:
        edges_set_idxs_list = ((0,), (1,))
    #     print (interSubgrsEdges)
        for idx, edges_set_idxs in enumerate(edges_set_idxs_list):
            for edges_set_idx in edges_set_idxs:
                for edge_lbl in interSubgrsEdges[edges_set_idx]:
    #                 print (edge_lbl)
                    if "/H" in edge_lbl or "/HEQ" in edge_lbl or  "/EQ" in edge_lbl: # and edge_lbl != "RSTR/H"?
    #                     if p:
    #                         print (idx, edges_set_idx, edge_lbl)
    #                     print (idx, "!")
                        if outest_scope != None:
                            if scopes[idx] and scopes[idx] != outest_scope:
    #                             print ("multiple ltop?", derivRule)
                                multi_ltop  =True
    #                             input()
                                break
                        elif scopes[idx] != None:
    #                         print (idx, scopes[idx])
                            outest_scope = scopes[idx]
                if multi_ltop: break
            if multi_ltop: break
        # determine new ltop if no handle edges are found"
        derivRule_orig = derivRule.split("&")[0].split("^")[0]
        if outest_scope == None:
            if derivRule_orig in erg_rules and "_crd" in derivRule_orig or "mrk-" in derivRule_orig:
                return ltop_aval, outest_scope
            elif derivRule_orig in half_erg_rules and "j-j_" in derivRule_orig\
                or "pp-pp" in derivRule_orig:
                 return ltop_aval, outest_scope
            elif erg_rule2hdness.get(derivRule_orig) == "B-lhd": outest_scope = scopes[-2]
            elif erg_rule2hdness.get(derivRule_orig) == "B-rhd": outest_scope = scopes[-1]
            elif erg_rule2hdness.get(derivRule_orig) == "U-hd": outest_scope = scopes[-1]
        # check incoming "/H or /HEQ" edges here (or use them to decide on new ltop)
        outest_scope_new = None
        for extEdgeKeyExts in extEdgeKeyExts_list:
            for key, ext in extEdgeKeyExts:
                if ext[0] == 't' and ("/H" in edgeKey2lbl[key]\
                                      or "/HEQ" in edgeKey2lbl[key] or "/EQ" in edgeKey2lbl[key]):
                    if outest_scope == None:
                        if outest_scope_new != None and outest_scope_new != dmrsNode2scope[key[1]]:
                            # multiple new ltop; treat as available
                            outest_scope_new = None
                            multi_ltop = True
                            break
                        else: outest_scope_new = dmrsNode2scope[key[1]]
                    elif outest_scope != None:
                        if dmrsNode2scope[key[1]] != outest_scope:
                            ltop_aval = False
    #                         print ("ltop not aval:", derivRule, key, edgeKey2lbl[key])
                            break
            if multi_ltop or not ltop_aval: break

        if outest_scope == None and outest_scope_new != None:
            outest_scope = outest_scope_new
        return ltop_aval, outest_scope

def get_ordered_external_nodes_repl(node_ind_subgraph, ordered_subgraphsNodes_list, dmrs_nxDG_repl, node_typed = False):
    # operate on dmrs_nxDG_repl to get external nodes from orig DMRS
    external_nodes_idx = []
    ordered_ext_nodes = []
    
    # for each subgraph in the merge
    for idx, subgraphNodes in enumerate(ordered_subgraphsNodes_list):
        # if originally ordered,
        # then check if each node inside node_repl is an external node; if yes, add to 'ordered_ext_nodes'
        for node_repl in subgraphNodes:
            
#             print (ordered_subgraphsNodes_list)
            try:
                orig_external_nodes = dmrs_nxDG_repl.nodes[node_repl]['ordered_ext_nodes'].split("&&")
            except:
                print ("get ord ext can't split")
#                 print (ordered_subgraphsNodes_list)
#                 print (dmrs_nxDG_repl.nodes())
#                 print (node_repl)
#                 print (dmrs_nxDG_repl.nodes[node_repl])
            all_edges_repl = set(list(dmrs_nxDG_repl.out_edges(node_repl, data='label'))
                                 + list(dmrs_nxDG_repl.in_edges(node_repl, data='label')))
            intraSubgr_edges_repl = set(list(node_ind_subgraph.out_edges(node_repl, data='label'))
                                       + list(node_ind_subgraph.in_edges(node_repl, data='label')))
            # retrieve the new external nodes by the ext_node info in the remaining edges
            # that connect to the outside
#             origAndNew_external_nodes = [n for n in orig_external_nodes for src, targ, lbl in (all_edges_repl - intraSubgr_edges_repl) if n in (src, targ)]
            orderAndNode = {(orig_external_nodes[get_src_ext(lbl, node_typed)], get_src_ext(lbl, node_typed)) if src == node_repl
                            else (orig_external_nodes[get_targ_ext(lbl, node_typed)], get_targ_ext(lbl, node_typed)) if targ == node_repl
                            else None
                            for src, targ, lbl in (all_edges_repl - intraSubgr_edges_repl)}
            if None in orderAndNode:
                print ("erroreous external nodes")
                util.write_figs_err(node_ind_subgraph, None, None)
                input()
            sorted_orderAndNode = sorted(orderAndNode, key = lambda t:t[1])
            if sorted_orderAndNode:
                ordered_ext_nodes += list(zip(*sorted_orderAndNode))[0]
#             for src, targ, lbl in all_edges_repl - intraSubgr_edges_repl:
#                 if src == node_repl:
#                     ext_node_tmp = orig_external_nodes[int(lbl.split("-src-")[0].split("#")[1])]
#                     if ext_node_tmp not in ordered_ext_nodes:
#                         ordered_ext_nodes.append(ext_node_tmp)
#                 elif targ == node_repl:
#                     ext_node_tmp = orig_external_nodes[int(lbl.split("-targ-")[1].split("#")[1])]
#                     if ext_node_tmp not in ordered_ext_nodes:
#                         ordered_ext_nodes.append(ext_node_tmp)
#     print (ordered_ext_nodes)
       
            

    return ordered_ext_nodes

 
def get_coarse_preTerm(entity):
    if "/" in entity and not is_syn_construction(entity):
        # print (entity.split("/", 1)[1])
        return entity.split("/", 1)[1]
    else:
        return entity
    
def is_derivNode_preTerm(deriv_nxDG, node):
    out_edges = list(deriv_nxDG.out_edges(node))
    if out_edges:
        dgtr = out_edges[0][1]
        return deriv_nxDG.nodes[dgtr]['cat'] == '<terminal>'
    else:
        return None

def get_surface_of_derivSubTree(deriv_nxDG, curr_node):
    def _dfs(curr_node):
        nonlocal surface
        if deriv_nxDG.nodes[curr_node]['cat'] == '<terminal>':
            surface += deriv_nxDG.nodes[curr_node]['form']
        dgtrs = get_deriv_leftRight_dgtrs(deriv_nxDG, curr_node)
        for dgtr in dgtrs:
            _dfs(dgtr)
    surface = ""
    _dfs(curr_node)
    return surface

def extract_semEmts_subtree(anno_deriv_nxDG_uni, annoDerivNode2dmrsNodeSubgr, annoPretermNode2canon_usp,
                           curr_node, matched_semEmts, semEmt_trgs, dmrs_nxDG_repl, dmrs_nxDG,
                            semEmtType = "semEmt"):
    invalid_tree = False
    def _dfs(curr_node, par_node, lr, path):
        nonlocal invalid_tree
        semEmt_key = semEmtType
        semEmt_trg_key = "{}_trg".format(semEmtType)
        semEmt_trgL, semEmt_trgR = semEmt_trgs
        semEmt_trg = semEmt_trgL or semEmt_trgR
        nonlocal subtree
        if anno_deriv_nxDG_uni.nodes[curr_node]['cat'] == '<terminal>': return
        curr_entity = anno_deriv_nxDG_uni.nodes[curr_node]['entity']
        add_entity = curr_entity
        if curr_node in annoPretermNode2canon_usp:
            add_entity = annoPretermNode2canon_usp[curr_node]
        subtree.add_node(curr_node, entity = add_entity)
        dgtrs = get_deriv_leftRight_dgtrs(anno_deriv_nxDG_uni, curr_node)
        curr_dmrs_nodes = annoDerivNode2dmrsNodeSubgr.get(curr_node) or []
        dgtrs_dmrs_nodes = [annoDerivNode2dmrsNodeSubgr.get(dgtr) or [] for dgtr in dgtrs]
        dgtrs_semEmt = get_dgtrsSemEmt(anno_deriv_nxDG_uni, curr_node, semEmtType)
        if par_node:
            subtree.add_edge(par_node, curr_node, label = lr)
        
        if len(dgtrs) == 1:
            if (not anno_deriv_nxDG_uni.nodes[dgtrs[0]]['cat'] == '<terminal>'\
                and not dgtrs_dmrs_nodes[0])\
                or dgtrs_semEmt\
                or not par_node:
                _dfs(dgtrs[0], curr_node, 'U', path + "U")
            elif dgtrs_dmrs_nodes[0]\
                or (curr_dmrs_nodes and anno_deriv_nxDG_uni.nodes[dgtrs[0]]['cat'] == '<terminal>'):
                    erg_cfg_dgtrs2path[add_entity] = path
                    erg_cfg_dgtr2cnt[add_entity] += 1
                    subtree.graph['dgtrs'].append(curr_node)
                    ent2node[add_entity].append(curr_node)
            elif (not curr_dmrs_nodes and anno_deriv_nxDG_uni.nodes[dgtrs[0]]['cat'] == '<terminal>'):
                if curr_entity not in matched_semEmts:
                    pass
#                     print ("for {}, extra semEmt on the way:".format(matched_semEmts), curr_entity)
#                     input()
                    
        elif len(dgtrs) == 2:
            if curr_dmrs_nodes and not(bool(dgtrs_dmrs_nodes[0]) and bool (dgtrs_dmrs_nodes[1]))\
                or not par_node:
                _dfs(dgtrs[0], curr_node, 'L', path + "L")
                _dfs(dgtrs[1], curr_node, 'R', path + "R")
            elif curr_dmrs_nodes:
                erg_cfg_dgtrs2path[add_entity] = path
                erg_cfg_dgtr2cnt[add_entity] += 1
                subtree.graph['dgtrs'].append(curr_node) 
                ent2node[add_entity].append(curr_node)
        if not par_node:
            if semEmt_trg_key in anno_deriv_nxDG_uni.nodes[curr_node]:
                if anno_deriv_nxDG_uni.nodes[curr_node][semEmt_trg_key] == semEmt_trg:
                    anno_deriv_nxDG_uni.nodes[curr_node][semEmt_trg_key] = None
#             else:
#                 print (anno_deriv_nxDG_uni.nodes[curr_node][semEmt_trg_key] == semEmt_trg)
            
#             print (anno_deriv_nxDG_uni.nodes[curr_node][semEmt_key], matched_semEmts[0])
            anno_deriv_nxDG_uni.nodes[curr_node][semEmt_key] = []
            
#             print (curr_node, semEmt_trg_key, "cleaned:", semEmt_trg)
            
    
    subtree = nx.DiGraph()
    subtree.graph['dgtrs'] = [] # used to store the node id of the connecting dgtrs of the subtree
    erg_cfg_dgtrs2path = {}
    erg_cfg_dgtr2cnt = Counter()
    left_cfg_dgtr_ent = None
    right_cfg_dgtr_ent = None
    ent2node = defaultdict(list)

    _dfs(curr_node, None, "", "")
    has_matched_semEmt = False
    for node, node_prop in subtree.nodes(data = True):
        if node not in subtree.graph['dgtrs'] and isinstance(node_prop['entity'], tuple):
            return None, None, None, None, None
        if node_prop['entity'] in matched_semEmts: has_matched_semEmt = True
    if not matched_semEmts: return None, None, None, None, None
        
    num_cfg_dgtrs = sum(erg_cfg_dgtr2cnt.values())
    if num_cfg_dgtrs == 2:
        if len(erg_cfg_dgtrs2path) == 1:
            print (erg_cfg_dgtr2cnt.keys(), "same dgtr entities for subtree")
            return subtree, list(erg_cfg_dgtr2cnt.keys())[0], list(erg_cfg_dgtr2cnt.keys())[0],\
                ent2node[list(erg_cfg_dgtr2cnt.keys())[0]][0], ent2node[list(erg_cfg_dgtr2cnt.keys())[0]][1]
        entities = list(erg_cfg_dgtrs2path.keys())
        paths = (erg_cfg_dgtrs2path[entities[0]], erg_cfg_dgtrs2path[entities[1]])
        diff_index = next((i for i in range(min(len(paths[0]),
                                               len(paths[1])))
                           if paths[0][i]!=paths[1][i]),
                          None)
        if paths[0][diff_index] < paths[1][diff_index]: left_cfg_dgtr_ent, right_cfg_dgtr_ent = entities
        elif paths[0][diff_index] > paths[1][diff_index]: right_cfg_dgtr_ent, left_cfg_dgtr_ent = entities
    else:
        print ("{} erg cfg dgtr in subtree?".format(len(erg_cfg_dgtrs2path)))
        print (erg_cfg_dgtrs2path)
#         write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, None, name = "28")
#         write_figs_err(dmrs_nxDG, subtree, None, name = "29")
        return None, None, None, None, None
    return subtree, left_cfg_dgtr_ent, right_cfg_dgtr_ent,\
        ent2node[left_cfg_dgtr_ent][0], ent2node[right_cfg_dgtr_ent][0]

    
# def extract_semEmt_subtree(anno_deriv_nxDG_uni, annoDerivNode2dmrsNodeSubgr, annoPretermNode2canon_usp,
#                            curr_node, semEmt, dmrs_nxDG_repl, dmrs_nxDG, semEmtType = "semEmt"):
#     # print ("extract sub tree now!")
#     invalid_tree = False
#     def _dfs(curr_node, par_node, lr, path):
#         nonlocal invalid_tree
#         semEmt_trg_in_subtree_key = "{}_trg_in_subtree".format(semEmtType)
#         semEmt_in_subtree_key = "{}_in_subtree".format(semEmtType)
#         semEmt_key = semEmtType
#         semEmt_trg_key = "{}_trg".format(semEmtType)
#         nonlocal subtree
#         if anno_deriv_nxDG_uni.nodes[curr_node]['cat'] == '<terminal>':
#             return
#         add_entity = anno_deriv_nxDG_uni.nodes[curr_node]['entity']
# #         pprint (add_entity)
#         if curr_node in annoPretermNode2canon_usp:
#             add_entity = annoPretermNode2canon_usp[curr_node]
#         subtree.add_node(curr_node, entity = add_entity)
#         curr_semEmt_in_subtree = anno_deriv_nxDG_uni.nodes[curr_node][semEmt_in_subtree_key]
# #         curr_trg_in_subtree = anno_deriv_nxDG_uni.nodes[curr_node]['semEmt_trg_in_subtree']
# #         curr_semEmt = anno_deriv_nxDG_uni.nodes[curr_node]['semEmt']
# #         curr_trg = anno_deriv_nxDG_uni.nodes[curr_node]['semEmt_trg']
#         dgtrs = get_deriv_leftRight_dgtrs(anno_deriv_nxDG_uni, curr_node)
#         curr_dmrs_nodes = annoDerivNode2dmrsNodeSubgr.get(curr_node) or []
# #         print (curr_node, curr_dmrs_nodes)
#         dgtrs_dmrs_nodes = [annoDerivNode2dmrsNodeSubgr.get(dgtr) or [] for dgtr in dgtrs]
# #         print (curr_node, dgtrs_dmrs_nodes)
#         if par_node:
#             subtree.add_edge(par_node, curr_node, label = lr)
        
#         if len(dgtrs) == 1:
#             if (not anno_deriv_nxDG_uni.nodes[dgtrs[0]]['cat'] == '<terminal>'\
#                 and not dgtrs_dmrs_nodes[0])\
#                 or semEmt in curr_semEmt_in_subtree\
#                 or not par_node:
#                 _dfs(dgtrs[0], curr_node, 'U', path + "U")
#             elif dgtrs_dmrs_nodes[0]\
#                 or (curr_dmrs_nodes and anno_deriv_nxDG_uni.nodes[dgtrs[0]]['cat'] == '<terminal>'):
#                 erg_cfg_dgtrs2path[add_entity] = path
#                 erg_cfg_dgtr2cnt[add_entity] += 1
#                 subtree.graph['dgtrs'].append(curr_node)
#                 ent2node[add_entity].append(curr_node)
# #                 print (add_entity, path)
#         elif len(dgtrs) == 2:
# #             print (add_entity, dgtrs_dmrs_nodes)
# #             print (bool(curr_dmrs_nodes), bool(dgtrs_dmrs_nodes[0]) ^ bool (dgtrs_dmrs_nodes[1]))
#             if curr_dmrs_nodes and bool(dgtrs_dmrs_nodes[0]) ^ bool (dgtrs_dmrs_nodes[1])\
#                 or not curr_dmrs_nodes or not par_node:
#                 # print (curr_node,"go")
#                 _dfs(dgtrs[0], curr_node, 'L', path + "L")
#                 _dfs(dgtrs[1], curr_node, 'R', path + "R")
#             elif curr_dmrs_nodes:
#                 erg_cfg_dgtrs2path[add_entity] = path
#                 erg_cfg_dgtr2cnt[add_entity] += 1
#                 subtree.graph['dgtrs'].append(curr_node) 
#                 ent2node[add_entity].append(curr_node)
#         if not par_node:
#             for semEmt_attr in [semEmt_in_subtree_key]:
#                 anno_deriv_nxDG_uni.nodes[curr_node][semEmt_attr] = []
#             for semEmt_attr in [semEmt_key, semEmt_trg_key, semEmt_trg_in_subtree_key]:
#                 anno_deriv_nxDG_uni.nodes[curr_node][semEmt_attr] = None
                    
#     subtree = nx.DiGraph()
#     subtree.graph['dgtrs'] = [] # used to store the node id of the connecting dgtrs of the subtree
#     erg_cfg_dgtrs2path = {}
#     erg_cfg_dgtr2cnt = Counter()
#     left_cfg_dgtr_ent = None
#     right_cfg_dgtr_ent = None
#     ent2node = defaultdict(list)
# #     test_rep = (((None, derivL, None), (None, derivR, None)), 
# #     print ("----------------------------")
#     _dfs(curr_node, None, "", "")
#     for node, node_prop in subtree.nodes(data = True):
# #         print (subtree.graph['dgtrs'], node)
# #         print ( node_prop['entity'],  node_prop['entity'][0])
# #         print (node not in subtree.graph['dgtrs'] and isinstance(node_prop['entity'], tuple))
#         if node not in subtree.graph['dgtrs'] and isinstance(node_prop['entity'], tuple):
#             return None, None, None, None, None
#     num_cfg_dgtrs = sum(erg_cfg_dgtr2cnt.values())
# #     print (num_cfg_dgtrs)
#     if num_cfg_dgtrs == 2:
#         if len(erg_cfg_dgtrs2path) == 1:
#             print (erg_cfg_dgtr2cnt.keys(), "only one dgtr for subtree")
#             return subtree, list(erg_cfg_dgtr2cnt.keys())[0], list(erg_cfg_dgtr2cnt.keys())[0],\
#                 ent2node[list(erg_cfg_dgtr2cnt.keys())[0]][0], ent2node[list(erg_cfg_dgtr2cnt.keys())[0]][1]
#         entities = list(erg_cfg_dgtrs2path.keys())
#         paths = (erg_cfg_dgtrs2path[entities[0]], erg_cfg_dgtrs2path[entities[1]])
#         diff_index = next((i for i in range(min(len(paths[0]),
#                                                len(paths[1])))
#                            if paths[0][i]!=paths[1][i]),
#                           None)
#         if paths[0][diff_index] < paths[1][diff_index]: left_cfg_dgtr_ent, right_cfg_dgtr_ent = entities
#         elif paths[0][diff_index] > paths[1][diff_index]: right_cfg_dgtr_ent, left_cfg_dgtr_ent = entities
#     else:
#         print ("{} erg cfg dgtr in subtree?".format(len(erg_cfg_dgtrs2path)))
#         print (erg_cfg_dgtrs2path)
#         write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, None, name = "28")
#         write_figs_err(dmrs_nxDG, subtree, None, name = "29")
#         input()
#     return subtree, left_cfg_dgtr_ent, right_cfg_dgtr_ent,\
#         ent2node[left_cfg_dgtr_ent][0], ent2node[right_cfg_dgtr_ent][0]
    
def get_dgtrsSemEmt(deriv_nxDG, node, semEmtType = "semEmt"):
    dgtrs = get_deriv_leftRight_dgtrs(deriv_nxDG, node)
    if len(dgtrs) == 1: return deriv_nxDG.nodes[dgtrs[0]].get(semEmtType)
    elif len(dgtrs) == 2: return deriv_nxDG.nodes[dgtrs[0]].get(semEmtType)\
        + deriv_nxDG.nodes[dgtrs[1]].get(semEmtType)
    
def get_dgtrsSemEmtInSubtree(deriv_nxDG, node, semEmtType = "semEmt"):
    dgtrs = get_deriv_leftRight_dgtrs(deriv_nxDG, node)
    return [semEmt
            for i in range(len(dgtrs))
                if deriv_nxDG.nodes[dgtrs[i]].get('{}_in_subtree'.format(semEmtType))
                    for semEmt in deriv_nxDG.nodes[dgtrs[i]].get('{}_in_subtree'.format(semEmtType))
                   ]
def get_dgtrsSemEmtTrg(deriv_nxDG, node):
    # for simplicity, randomly select one trg
    lOrR = None
    dgtrs = get_deriv_leftRight_dgtrs(deriv_nxDG, node)
    returned = deriv_nxDG.nodes[dgtrs[0]].get('semEmt_trg')
    if len(dgtrs) == 1:
        if not returned: return None, lOrR
        else: return returned, lOrR
    elif len(dgtrs) == 2:
        if returned: return returned, "L"
        elif deriv_nxDG.nodes[dgtrs[1]].get('semEmt_trg'):
            return deriv_nxDG.nodes[dgtrs[1]].get('semEmt_trg'), "R"
        else: return None, lOrR

def get_dgtrsSemEmtTrgInSubtree(deriv_nxDG, node):
    # for simplicity, randomly select one trg
    lOrR = None
    dgtrs = get_deriv_leftRight_dgtrs(deriv_nxDG, node)
    returned = deriv_nxDG.nodes[dgtrs[0]].get('semEmt_trg_in_subtree')
    if len(dgtrs) == 1:
        if not returned: return None, lOrR
        else: return returned, lOrR
    elif len(dgtrs) == 2:
        if returned: return returned, "L"
        elif deriv_nxDG.nodes[dgtrs[1]].get('semEmt_trg_in_subtree'):
            return deriv_nxDG.nodes[dgtrs[1]].get('semEmt_trg_in_subtree'), "R"
        else: return None, lOrR
    
def get_dgtrsCopulaTrgInSubtree(deriv_nxDG, node):
    returned_trg = None
    lorR = None
    semEmtType = "copula"
#     out_edges = list(deriv_nxDG.out_edges(node))
    out_nbs = get_deriv_leftRight_dgtrs(deriv_nxDG, node)
    # check headedness
    if len(out_nbs) == 1:
        trg_subtree = deriv_nxDG.nodes[out_nbs[0]].get('{}_trg_in_subtree'.format(semEmtType))
        trg = deriv_nxDG.nodes[out_nbs[0]].get('{}_trg'.format(semEmtType))
        if trg: returned_trg = trg
        else: returned_trg = trg_subtree
    elif len(out_nbs) == 2:
        l_trg_subtree = deriv_nxDG.nodes[out_nbs[0]].get('{}_trg_in_subtree'.format(semEmtType))
        r_trg_subtree = deriv_nxDG.nodes[out_nbs[1]].get('{}_trg_in_subtree'.format(semEmtType))
        l_trg = deriv_nxDG.nodes[out_nbs[0]].get('{}_trg'.format(semEmtType))
        r_trg = deriv_nxDG.nodes[out_nbs[1]].get('{}_trg'.format(semEmtType))
        erg_rule = deriv_nxDG.nodes[node]['entity'].split("&")[0].split("^")[0]
        # print (erg_rule, erg_rule2hdness.get(erg_rule))
        # print (l_trg, r_trg, 'tree')
#         if node == 15719:
#             print (erg_rule2hdness.get(erg_rule), erg_rule)
        if erg_rule2hdness.get(erg_rule) == "B-lhd":
            if l_trg: returned_trg, lorR = l_trg, "L"
            elif l_trg_subtree: returned_trg, lorR = l_trg_subtree, "L"
            elif r_trg: returned_trg, lorR = r_trg, "R"
            elif r_trg_subtree: returned_trg, lorR = r_trg_subtree, "R"
            else: returned_trg, lorR = None, None
        else:
            if r_trg: returned_trg, lorR = r_trg, "R"
            elif r_trg_subtree: returned_trg, lorR = r_trg_subtree, "R"
            elif l_trg: returned_trg, lorR = l_trg, "L"
            elif l_trg_subtree: returned_trg, lorR = l_trg_subtree, "L"
            else: returned_trg, lorR = None, None
    return returned_trg, lorR

def get_dgtrsTrg(deriv_nxDG, node, dgtr_dmrs_nodes, semEmtType = "semEmt"):
    returned_trg = None
    lorR = None
#     out_edges = list(deriv_nxDG.out_edges(node))
    out_nbs = get_deriv_leftRight_dgtrs(deriv_nxDG, node)
    # check headedness
    if len(out_nbs) == 1:
        trg = deriv_nxDG.nodes[out_nbs[0]].get('{}_trg'.format(semEmtType))
        if trg: returned_trg = trg
    elif len(out_nbs) == 2:
        l_trg = deriv_nxDG.nodes[out_nbs[0]].get('{}_trg'.format(semEmtType))
        r_trg = deriv_nxDG.nodes[out_nbs[1]].get('{}_trg'.format(semEmtType))
        if not dgtr_dmrs_nodes[0]: return r_trg, "R"
        elif not dgtr_dmrs_nodes[1]: return l_trg, "L"
        erg_rule = deriv_nxDG.nodes[node]['entity'].split("&")[0].split("^")[0]
        if erg_rule2hdness.get(erg_rule) == "B-lhd":
            if l_trg: returned_trg, lorR = l_trg, "L"
            elif r_trg: returned_trg, lorR = r_trg, "R"
            else: returned_trg, lorR = None, None
        else:
            if r_trg: returned_trg, lorR = r_trg, "R"
            elif l_trg: returned_trg, lorR = l_trg, "L"
            else: returned_trg, lorR = None, None
    return returned_trg, lorR
    
def get_dgtrsTrg_test(trgs, erg_rule, semEmtType = None):
    if not trgs: returned_trg = None
    elif len(trgs) == 1:
        if trgs[0]: returned_trg = trg[0]
        else: returned_trg = None
    elif len(trgs) >= 2:
        l_trg, r_trg = trgs[-2:]
        ergRule_orig = erg_rule.split("&")[0].split("^")[0]
        if erg_rule2hdness.get(ergRule_orig) == "B-lhd":
            if l_trg: returned_trg = l_trg
            elif r_trg: returned_trg = r_trg
            else: returned_trg = None
        else:
            if r_trg: returned_trg = r_trg
            elif l_trg: returned_trg = l_trg
            else: returned_trg = None
    return returned_trg
    
def get_dgtrsCopulaTrg_test(trgs, erg_rule):
#     returned_trg None
    if not trgs: returned_trg = None
    elif len(trgs) == 1:
        if trgs[0]: returned_trg = trg[0]
        else: returned_trg = None
    elif len(trgs) >= 2:
        l_trg, r_trg = trgs[-2:]
        ergRule_orig = erg_rule.split("&")[0].split("^")[0]
        if erg_rule2hdness.get(ergRule_orig) == "B-lhd":
            if l_trg: returned_trg = l_trg
            elif r_trg: returned_trg = r_trg
            else: returned_trg = None
        else:
            if r_trg: returned_trg = r_trg
            elif l_trg: returned_trg = l_trg
            else: returned_trg = None
    return returned_trg
 
def get_dgtrsSemEmtTrg_test(trgs):
    if not trgs: return None
    elif len(trgs) == 1:
        if not trgs[0]: return None
        else: return trgs[0]
    elif len(trgs) >= 2:
        l_trg, r_trg = trgs[-2:]
        if l_trg: return l_trg
        elif r_trg: return r_trg
        else: return None
    
def is_node_both_semEmtAndTrg(deriv_nxDG, node, semEmtType = "semEmt"):
    return not (not deriv_nxDG.nodes[node].get('{}_trg_in_subtree'.format(semEmtType))\
                or not len(deriv_nxDG.nodes[node].get('{}_in_subtree'.format(semEmtType))) >= 1)

def get_predTrgSemEmt(dmrs_nxDG, interSubgrs_edges_key, deriv_nxDG, node):
    semEmtType = 'predSemEmt'
    dgtrs = get_deriv_leftRight_dgtrs(deriv_nxDG, node)
    semEmtsL = deriv_nxDG.nodes[dgtrs[0]][semEmtType]
    semEmtsR = deriv_nxDG.nodes[dgtrs[1]][semEmtType]
    predsAndSemEmts = []
#     idx2subgrIdx = {0: 0, 1:1, 2:0, 3:2, 4:1, 5:2}
    if semEmtsL or semEmtsR:
        for semEmt in semEmtsL + semEmtsR:
            semEmt_surf = semEmt.split("_")[0]
            semEmt_preterm_orig = semEmt.split("&")[0].split("/")[0]
            for idx, keys in enumerate(interSubgrs_edges_key):
                for src, targ in keys:
    #                 print (interSubgrs_edges_key)
    #                 print (src)
                    src_pred = dmrs_nxDG.nodes[int(src)]['instance']
    #                 targ_pred = dmrs_nxDG.nodes[targ]['instance']
                    if src_pred in trgPred2semEmt:
                        if semEmt_surf in trgPred2semEmt[src_pred]\
                            or semEmt_preterm_orig in trgPred2semEmt[src_pred]:
                            interSubgrs_edge_lbl = dmrs_nxDG.edges[int(src), int(targ), 0]['label']
                            predsAndSemEmts.append((src_pred, semEmt_preterm_orig, interSubgrs_edge_lbl, idx))
#                 if targ_pred in predSemEmts:
#                     if semEmt_surf in trgPred2semEmt[targ_pred] or semEmt_preterm_orig in trgPred2semEmt[targ_pred]:
#                         extractables.append((targ_pred, semEmt_preterm_orig))
    return predsAndSemEmts

def is_semEmt_extractable(deriv_nxDG, node, semEmtType = "semEmt"):
    dgtrs = get_deriv_leftRight_dgtrs(deriv_nxDG, node)
    if semEmtType != 'by':
        semEmt_trgL = deriv_nxDG.nodes[dgtrs[0]][f'{semEmtType}_trg']
        semEmt_trgR = deriv_nxDG.nodes[dgtrs[1]][f'{semEmtType}_trg']
    semEmtsL = deriv_nxDG.nodes[dgtrs[0]][semEmtType]
    semEmtsR = deriv_nxDG.nodes[dgtrs[1]][semEmtType]
    if semEmtType == 'by':
        targ_semEmt_trg =  deriv_nxDG.nodes[dgtrs[0]]['copula_trg']
        if targ_semEmt_trg and "psv:PSV" in targ_semEmt_trg\
            and any(["by_pass_p" in s for s in semEmtsL + semEmtsR]):
            return ['by_pass_p'], targ_semEmt_trg, None
        elif targ_semEmt_trg and "nom:+" in targ_semEmt_trg\
            and any(["of_prtcl" in s for s in semEmtsL + semEmtsR]):
            return ['of_prtcl'], targ_semEmt_trg, None
        else: return [], None, None
    
    if semEmtType == 'copula':
        matched_semEmts = []
        targ_semEmt_trgs = [semEmt_trgR, semEmt_trgL]
#         print (semEmtsL, semEmtsR, targ_semEmt_trg)
        if (semEmtsL or semEmtsR) and semEmt_trgL or semEmt_trgR:
            for idx, targ_semEmt_trg in enumerate(targ_semEmt_trgs):
                if not targ_semEmt_trg: continue
                for semEmt in semEmtsL + semEmtsR:
                    semEmt_orig = semEmt.split("&")[0].split("/")[0]
    #                 print (semEmt_orig)
                    if semEmt_orig in copula2trg:
                        for listOfProps in copula2trg[semEmt_orig]:
#                             print (listOfProps, semEmt_orig, targ_semEmt_trg)
                            if all([keyValue in targ_semEmt_trg for keyValue in listOfProps]):
                                if semEmt_orig in ['do1_pos', 'did1_pos', 'does1_pos']:
                                    if "neg:-" in targ_semEmt_trg or "sf:Q" in targ_semEmt_trg:
                                        matched_semEmts.append(semEmt_orig)
                                else:
                                    matched_semEmts.append(semEmt_orig)
                                    break
                if matched_semEmts:
                    if idx == 0: return matched_semEmts, None, targ_semEmt_trg
                    elif idx == 1: return matched_semEmts, targ_semEmt_trg, None
        return matched_semEmts, None, None
    elif semEmtType == 'prtcl':
        matched_semEmts = []
        targ_semEmt_trgs = [semEmt_trgL, semEmt_trgR]
#         print (semEmtType, semEmtsL, dgtrs[0], semEmtsR, dgtrs[1])
#         print ("prtcl trgs: ", targ_semEmt_trgs)
        if (semEmtsL or semEmtsR) and semEmt_trgL or semEmt_trgR:
            for idx, targ_semEmt_trg in enumerate(targ_semEmt_trgs):
                if not targ_semEmt_trg: continue
                for semEmt in semEmtsL + semEmtsR:
                    semEmt_orig = semEmt.split("&")[0].split("/")[0]
                    if is_prtclInTrg(semEmt_orig, targ_semEmt_trg):
                        matched_semEmts.append(semEmt_orig)
                if matched_semEmts:
                    if idx == 0: return matched_semEmts, targ_semEmt_trg[0], None
                    elif idx == 1: return matched_semEmts, None, targ_semEmt_trg[1]
        return matched_semEmts, None, None
    if semEmtType == 'compl':
        matched_semEmts = []
        targ_semEmt_trg = semEmt_trgR
        if (semEmtsL or semEmtsR) and targ_semEmt_trg:
            for semEmt in semEmtsL + semEmtsR:
                semEmt_orig = semEmt.split("&")[0].split("/")[0]
                if semEmt_orig in compl2trg:
                    for listOfProps in compl2trg[semEmt_orig]:
                        if all([keyValue in targ_semEmt_trg for keyValue in listOfProps]):
                            matched_semEmts.append(semEmt_orig)
        return matched_semEmts, None, semEmt_trgR
        
        
def is_prtclInTrg(semEmt, trg):
    canon, pred, sense = trg
    semEmt_surface = semEmt.split("_")[0]
    if pred[0] == "_":
        sense = re.sub('-|\+', ' ', sense)
        if sense.isnumeric():
            return False
        return sense == semEmt_surface or semEmt_surface in sense.split(" ")
    return False
        
def is_semEmtsInTrg(semEmt, trg):
    canon, pred, sense = trg
    semEmt_surface = semEmt.split("_")[0]
    if pred[0] == "_":
        sense = re.sub('-|\+', ' ', sense)
        if sense.isnumeric():
            return False
        return sense == semEmt_surface or semEmt_surface in sense.split(" ")
    else:
        # print (semEmt,  trgPred2semEmt[pred], pred)
        return semEmt_surface == trgPred2semEmt[pred]
    
def is_copula(semEmt):
    if semEmt.startswith("be_c"): return True
    if semEmt == 'to_c_prop': return True
    semEmt_surface = semEmt.split("_")[0]
    if semEmt_surface in copula: return True
    if semEmt_surface in ['do1', 'did1', 'does1']: return True
    return False
def is_prtcl(semEmt):
    preterm_orig = semEmt.split("&")[0].split("/")[0]
    surface = preterm_orig.split("_")[0]
    return surface in particles and (preterm_orig[-8:] == "particle" or preterm_orig[-5:] == "prtcl")
def is_compl(semEmt):
    preterm_orig = semEmt.split("&")[0].split("/")[0]
    return preterm_orig in compl2trg
def is_predSemEmt(semEmt):
    preterm_orig = semEmt.split("&")[0].split("/")[0]
    semEmt_surface = semEmt.split("_")[0]
    return semEmt_surface in predSemEmts or preterm_orig == 'who2'
def is_by(semEmt):
    preterm_orig = semEmt.split("&")[0].split("/")[0]
#     semEmt_surface = semEmt.split("_")[0]
    return preterm_orig in byOf
    

def get_common_trg(dmrs_nxDG, node):
    pred_trg = None
    prop = dmrs_nxDG.nodes[node]
    pred = prop.get('instance')
    if pred in trgPred2semEmt:
        return trgPred2semEmt[pred]
        pred_trg = pred
        
def get_compl_trg(dmrs_nxDG, node):
    compl_trg = None
    prop = dmrs_nxDG.nodes[node]
    if prop.get('cvarsort') != 'e':
        return None
    if prop['tense'] == "UNTENSED":
        return None
    if prop['instance'][0] == "_":
         compl_trg = node_toString(node,prop,dmrs_nxDG,underspecLemma = True,underspecCarg=True,
                                   add_posSpecInfo = False)
    return compl_trg

def get_prtcl_trg(dmrs_nxDG, node, sense):
    pred_trg = None
    prop = dmrs_nxDG.nodes[node]
    if prop['instance'][0] == "_":
        pred_lemma, pred_pos = get_lemma_pos(prop)
        pred_trg = "ep:" + prop['instance'] + ';'
        pred_trg += "prtcl:" + sense
        if pred_pos == 'v' or unknown2pos[pred_pos] == 'v':
            # consider passive voice
            psv = "ACT"
            in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
            for in_edge in in_edges:
                edge_label = in_edge[2]['label']
                edge_source_pred = dmrs_nxDG.nodes[in_edge[0]]['instance']
                if edge_label == "ARG1/EQ" and edge_source_pred == 'parg_d':
                    psv = "PSV"
                    break
            pred_trg += "@" + "psv:" + psv
    return pred_trg
        
    
def get_copula_trg(dmrs_nxDG, node):
    prop = dmrs_nxDG.nodes[node]
    # print (prop)
    if prop.get('cvarsort') != 'e':
        return None, None
    if prop.get('instance') == 'parg_d': return None, None
    copula_trg = copula_trg_node_prop_toString(prop)
    if "pos:a" in copula_trg:
        arg1_cvarsort = None
        arg1_eqneq = None
        out_edges = dmrs_nxDG.out_edges(nbunch=[node], data='label')
        for src, targ, lbl in out_edges:
            if "ARG1" in lbl:
                arg1_cvarsort = dmrs_nxDG.nodes[targ].get('cvarsort')
                if "ARG1/EQ" in lbl: arg1_eqneq = "EQ"
                elif "ARG1/NEQ" in lbl: arg1_eqneq = "NEQ"
                break
        if not arg1_cvarsort or arg1_cvarsort != 'x':
            return None, None
        if arg1_eqneq != "NEQ" and arg1_eqneq == "EQ" and "tn:UT" in copula_trg:
            return None, None
    # get arg123's pers
    args_persNum = [None, None, None]
    out_edges = dmrs_nxDG.out_edges(nbunch=[node],data=True)
    for out_edge in out_edges:
        edge_label = out_edge[2]['label'].split("/")[0]
        if edge_label.startswith("ARG") and len(edge_label)==4:
            # print (out_edge[2]['label'], dmrs_nxDG.nodes[node], dmrs_nxDG.nodes[out_edge[1]])
            edge_type, arg_no = (edge_label[:3], int(edge_label[3:4]))
            if arg_no > 3:
                continue
            pers = dmrs_nxDG.nodes[out_edge[1]].get('pers') or "-P"
            num = dmrs_nxDG.nodes[out_edge[1]].get('num') or "-N"
            args_persNum[arg_no-1] = "&".join([num, pers])
    for arg_no, persnum in enumerate(args_persNum):
        if persnum:
            copula_trg += "@" + "arg" + str(arg_no+1) + ":" + persnum
    # consider passive voice
    psv = "ACT"
    in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
    for in_edge in in_edges:
        edge_label = in_edge[2]['label']
        edge_source_pred = dmrs_nxDG.nodes[in_edge[0]]['instance']
        if edge_label == "ARG1/EQ" and edge_source_pred == 'parg_d':
            psv = "PSV"
            break
    copula_trg += "@" + "psv:" + psv
    # consider 3sg for tn:PR
    sg3 = "-"
    if "tn:PR" in copula_trg and "sf:P" in copula_trg:
        for src, targ, lbl in dmrs_nxDG.out_edges(node, data = 'label'):
            if "ARG1" in lbl and psv == "ACT" or  "ARG2" in lbl and psv == "PSV":
                if str(dmrs_nxDG.nodes[targ].get('pers')) == '3' and dmrs_nxDG.nodes[targ].get('num') == 'SG':
                    sg3 = "+"
    copula_trg += "@" + "3sg:" + sg3
    # consider nominalization (for of_prtcl)
    nom = "-"
    in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
    for in_edge in in_edges:
        edge_label = in_edge[2]['label']
        edge_source_pred = dmrs_nxDG.nodes[in_edge[0]]['instance']
        if edge_label == "ARG1/HEQ" and edge_source_pred == 'nominalization':
            nom = "+"
            break
    copula_trg += "@" + "nom:" + nom
    # consider negation
    neg = "+"
    in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
    for in_edge in in_edges:
        edge_label = in_edge[2]['label']
        edge_source_pred = dmrs_nxDG.nodes[in_edge[0]]['instance']
        if edge_label == "ARG1/H" and edge_source_pred == 'neg':
            neg = "-"
            break
    copula_trg += "@" + "neg:" + neg
    # consider arg1 if _*_a_*
    if "pos:a" in copula_trg and arg1_eqneq:
        copula_trg += "@" + "a1eq:" + arg1_eqneq
        # print (copula_trg)
    return copula_trg, prop.get('tense')
    
def is_copulasInTrg(copula, trg):
    copula = copula.split("&")[0].split("/")[0]
    if copula == 'to_c_prop':
        return True
    if copula in copula2trg:
        for listOfProps in copula2trg[copula]:
            if all([keyValue in trg for keyValue in listOfProps]):
                if copula in ['do1_pos', 'did1_pos', 'does1_pos']:
                    if "neg:-" in trg or "sf:Q" in trg:
                        return True
                else:
                    return True
    return False

def propagate_semEmtInfo_test(derivRule, semEmtTrgs):
    copula_trgs, prtcl_trgs, compl_trgs = tuple(zip(*semEmtTrgs))
    new_copTrg, new_prtclTrg, new_complTrg = (None, None, None)
    new_copTrg  = get_dgtrsTrg_test(copula_trgs[-2:], derivRule)
    new_prtclTrg = get_dgtrsTrg_test(prtcl_trgs[-2:], derivRule)
    new_complTrg = get_dgtrsTrg_test(compl_trgs[-2:], derivRule)
    return new_copTrg, new_prtclTrg, new_complTrg

def propagate_semEmt_in_deriv(anno_deriv_nxDG_uni, curr_node, dgtr_dmrs_nodes):
    semEmt_types = ["copula", "prtcl", "compl", "by", "predSemEmt"]
    for semEmt in semEmt_types:
        if semEmt in ['copula', 'prtcl', 'compl']:
            semEmt_trg_key = "{}_trg".format(semEmt)
            trgInDgtr, lOrR = get_dgtrsTrg(anno_deriv_nxDG_uni, curr_node, dgtr_dmrs_nodes, semEmtType = semEmt)
#             print (curr_node, semEmt, trgInDgtr)
            anno_deriv_nxDG_uni.nodes[curr_node][semEmt_trg_key] = trgInDgtr
        semEmtInDgtr = get_dgtrsSemEmt(anno_deriv_nxDG_uni, curr_node, semEmtType = semEmt)
        anno_deriv_nxDG_uni.nodes[curr_node][semEmt] = semEmtInDgtr
    

def propagate_semEmtInfo_in_deriv(anno_deriv_nxDG_uni, curr_node, semEmtType = "semEmt"):
    semEmt_in_subtree_key = "{}_in_subtree".format(semEmtType)
    semEmt_trg_in_subtree_key = "{}_trg_in_subtree".format(semEmtType)
    semEmt_key = semEmtType
    semEmt_trg_key = "{}_trg".format(semEmtType)
    
    semEmtInDgtr = get_dgtrsSemEmt(anno_deriv_nxDG_uni, curr_node, semEmtType)
    semEmtInSubTree = get_dgtrsSemEmtInSubtree(anno_deriv_nxDG_uni, curr_node, semEmtType)
    
        
    # print (semEmtType, curr_node, semEmtInDgtr, semEmtInSubTree)
    if semEmtInDgtr or semEmtInSubTree:
        # print (semEmtInDgtr, semEmtInSubTree)
        anno_deriv_nxDG_uni.nodes[curr_node][semEmt_in_subtree_key] += semEmtInDgtr
        anno_deriv_nxDG_uni.nodes[curr_node][semEmt_in_subtree_key] += semEmtInSubTree
        
    if semEmtType == 'copula':
        copulaInDgtr, lOrR = get_dgtrsCopulaTrgInSubtree(anno_deriv_nxDG_uni, curr_node)
        anno_deriv_nxDG_uni.nodes[curr_node][semEmt_trg_in_subtree_key] = copulaInDgtr
        return lOrR
        
    elif semEmtType == 'semEmt':
        lOrR = None
        semEmtTrgInDgtr, lOrR1 = get_dgtrsSemEmtTrg(anno_deriv_nxDG_uni, curr_node)
        semEmtTrgInSubTree, lOrR2 = get_dgtrsSemEmtTrgInSubtree(anno_deriv_nxDG_uni, curr_node)
        if semEmtTrgInDgtr:
    #         print (semEmtTrgInDgtr)
            anno_deriv_nxDG_uni.nodes[curr_node][semEmt_trg_in_subtree_key] = semEmtTrgInDgtr
            lOrR = lOrR1
        elif semEmtTrgInSubTree:
    #         print (semEmtTrgInSubTree, "subbbb")
            anno_deriv_nxDG_uni.nodes[curr_node][semEmt_trg_in_subtree_key] = semEmtTrgInSubTree
            lOrR = lOrR2
        return lOrR
#     if semEmtInDgtr or semEmtInSubTree or semEmtTrgInDgtr or semEmtTrgInSubTree:
#         draw = input()
#         if draw:
#             util.write_figs_err(dmrs_nxDG_repl, anno_deriv_nxDG_uni, sentence)
#             input()
#         pass
    # print (curr_node, "copula_in_subtree", anno_deriv_nxDG_uni.nodes[curr_node]["copula_in_subtree"])

def add_semEmtTrgAttr2deriv(dmrs_nxDG, deriv_nxDG, curr_deriv_node, dmrs_nodes, canon):
    for node in dmrs_nodes:
        semEmt = get_pred_prtcl(dmrs_nxDG.nodes[node]['instance'])
        # copula trg
        copula_trg, tense = get_copula_trg(dmrs_nxDG, node)
        if (not deriv_nxDG.nodes[curr_deriv_node]['copula_trg']\
            or tense != 'UNTENSED') and copula_trg:
            deriv_nxDG.nodes[curr_deriv_node]['copula_trg'] = copula_trg
        # general semEmt trg
        elif semEmt:
            deriv_nxDG.nodes[curr_deriv_node]['semEmt_trg'] = (canon,
                                                              dmrs_nxDG.nodes[node]['instance'],
                                                              semEmt)
            break
            
def add_canonAttr2deriv(deriv_nxDG, curr_deriv_node, canon):
    deriv_nxDG.nodes[curr_deriv_node]['surfaceCanon'] = canon

def get_surfaceFromCanon(preTermCanon2surface2cnt, preTermCanon2bTagSurface2cnt, surface_canon, add_surf_prop):
    surface = None
    if len(surface_canon) >= 2:
        if preTermCanon2bTagSurface2cnt.get(surface_canon):
            surface = preTermCanon2bTagSurface2cnt.get(surface_canon).most_common()[0][0]
        elif preTermCanon2surface2cnt.get(surface_canon):
            surface = preTermCanon2surface2cnt.get(surface_canon).most_common()[0][0]
        else:
            surface = ""
            for node in surface_canon:
                pred = node[0].split("ep:")[1].split(";")[0]
                if pred[0] == "_":
                    _, pred_lemma, pred_pos, *_ = props['ep'].split("_")
                    surface += pred_lemma + " "
            surface = surface.strip()
#         pass
        
    elif len(surface_canon) == 1:
        prop_str = surface_canon[0][0]
        prop_list = prop_str.split(";")
        props = {prop.split(":")[0]: prop.split(":")[1] for prop in prop_list}
        cand_surface, cand_cnt = None, 0
        if preTermCanon2surface2cnt.get(surface_canon):
            cand_surface, cand_cnt = preTermCanon2surface2cnt.get(surface_canon).most_common()[0]
        if props['ep'] == 'pron':
#             print (add_surf_prop, cand_surface)
            if not cand_surface: surface = ""
            elif add_surf_prop == "subj" and not cand_surface.lower() in subj2objPron:
                surface = obj2subjPron.get(cand_surface.lower())
            elif add_surf_prop == "obj" and not cand_surface.lower() in obj2subjPron:
                surface = subj2objPron.get(cand_surface.lower())
#                 print (add_surf_prop, cand_surface, surface)
            if not surface: surface = cand_surface
        elif props['ep'].startswith("_do_") and props.get('tn') == 'PR'\
            and props.get('pf') == "-" and props.get('pg') == "-":
            if not add_surf_prop or 'neg:-' in add_surf_prop: surface = 'do'
            elif 'ps:3' in add_surf_prop and 'nm:SG' in add_surf_prop: surface = 'does'
            else: surface = 'do'
        elif props['ep'].startswith("_have_") and props.get('tn') == 'PR'\
            and props.get('pf') == "-" and props.get('pg') == "-":
            if not add_surf_prop or 'neg:-' in add_surf_prop: surface = 'have'
            elif 'ps:3' in add_surf_prop and 'nm:SG' in add_surf_prop: surface = 'has'
            else: surface = 'have'
        elif props['ep'].startswith("_be_") and props.get('tn') in ['PR', 'PA']\
            and props.get('pf') == "-" and props.get('pg') == "-":
            if not add_surf_prop or 'neg:-' in add_surf_prop: surface = 'be'
            elif props.get('tn') == 'PR':
                if 'ps:3' in add_surf_prop and 'nm:SG' in add_surf_prop: surface = 'is'
                elif 'ps:2' in add_surf_prop or 'nm:PL' in add_surf_prop: surface = 'are'
                elif 'ps:1' in add_surf_prop and 'nm:SG' in add_surf_prop: surface = 'am'
            elif props.get('tn') == 'PA':
                if 'ps:2' in add_surf_prop or 'nm:PL' in add_surf_prop: surface = 'were'
                elif 'ps:3' in add_surf_prop or 'ps:1' in add_surf_prop: surface = 'was'
            else: surface = 'be'
        elif props.get('cg'):
            if cand_cnt > 3:
                surface = cand_surface
            else:
                surface = props.get('cg').replace("+", " ")
        elif cand_surface: # and not add_surf_prop == "ps:3;nm:SG":
            surface = cand_surface
        elif props['ep'][0] == '_':
#             print (cand_surface, props)
            _, pred_lemma, pred_pos, *_ = props['ep'].split("_")
            pred_lemma_surf = pred_lemma.replace("+", " ")
            if pred_pos == 'n' or unknown2pos.get(pred_pos) == 'n':
                num = props.get('num')
                if num == 'SG':
                    surface = pred_lemma_surf
                elif num == 'PL':
                    if pred_lemma_surf.endswith(('s', 'sh', 'h', 'x', 'z', 'o')):
                        surface = pred_lemma_surf + 'es'
                    elif pred_lemma_surf.endswith(('f')):
                        surface = pred_lemma_surf[0:-1] + 'ves'
                    elif pred_lemma_surf.endswith(('fe')):
                        surface = pred_lemma_surf[0:-2] + 'ves'
                    elif pred_lemma_surf.endswith('y') and not pred_lemma_surf[-2] in VOWELS:
                        surface = pred_lemma_surf[0:-1] + 'ies'
                    elif pred_lemma_surf.endswith('y') and not pred_lemma_surf[-2] in VOWELS:
                        surface = pred_lemma_surf[0:-1] + 'ies'
                    else:
                        surface = pred_lemma_surf + 's'
                else:
                    surface = pred_lemma_surf
            elif pred_pos == 'v' or unknown2pos.get(pred_pos) == 'v':
                if add_surf_prop == "ps:3;nm:SG;tn:PR":
                    if pred_pos == 'v':
                        if pred_lemma_surf.endswith(('s', 'sh', 'h', 'x', 'z', 'o')):
                            surface = pred_lemma_surf + 'es'
                        elif pred_lemma_surf.endswith(('f')):
                            surface = pred_lemma_surf[0:-1] + 'ves'
                        elif pred_lemma_surf.endswith(('fe')):
                            surface = pred_lemma_surf[0:-2] + 'ves'
                        elif pred_lemma_surf.endswith('y') and not pred_lemma_surf[-2] in VOWELS:
                            surface = pred_lemma_surf[0:-1] + 'ies'
                        elif pred_lemma_surf.endswith('y') and not pred_lemma_surf[-2] in VOWELS:
                            surface = pred_lemma_surf[0:-1] + 'ies'
                        else: surface = pred_lemma_surf + 's'
                    else:
                        surface = pred_lemma_surf + 's'
                else:
                    t = props.get('tn')
                    r = props.get('pf')
                    p = props.get('pg')
                    verb_form2suffix = {("PA","-","-"):"ed", ("PA", "+", "-"):"ed"} # 3sg/"" not handled
                    if pred_pos == 'v':
                        if pred_lemma_surf[-1] == 'e':
                            if verb_form2suffix.get((t,r,p))\
                                and (not add_surf_prop or 'neg:-' not in add_surf_prop):
                                surface = pred_lemma_surf[:-1] + verb_form2suffix[(t,r,p)]
                            elif p == '+':
                                surface = pred_lemma_surf[:-1] + "ing"
                            else:
                                surface = pred_lemma_surf
                        else:
                            if verb_form2suffix.get((t,r,p)):
                                surface = pred_lemma_surf + verb_form2suffix[(t,r,p)]
                            elif p == '+':
                                surface = pred_lemma_surf + "ing"
                            else:
                                surface = pred_lemma_surf
                    else: surface = pred_lemma_surf
            elif pred_pos == 'a' or unknown2pos.get(pred_pos) == 'a':
                a, cs = props.get('a1cv'), props.get('cOrs')
                # adj case (unknown case of i)
                if a in ['x','i']:
                    surface = pred_lemma_surf
                    # handle comp/superl
#                     if cs == 'C':
#                         surface = pred_lemma_surf + "er"
#                     elif cs == 'S':
#                         surface = pred_lemma_surf + "est"
                # adv case
                elif a == 'e':
                    if "+" not in pred_lemma and pred_lemma_surf[-2:] != 'ly' and pred_pos == 'a':
                        surface = pred_lemma_surf # + "ly"
                    else:
                        surface = pred_lemma_surf
                    # don't handle comp/superl (assume most of them exist as more/most <lemma>-ly)
                else:
                    surface = pred_lemma_surf
            else:
                surface = pred_lemma_surf
        
        elif props['ep'][0] != '_':
            surface = ""
        
            
    return surface

     
   


    
def assign_node_key(node,node_prop,dmrs_nxDG,underspecLemma = False,underspecProps = False,usage = 'surface'):
    # for "_*" predicates
    if node_prop['instance'][0] == '_':
        pred_lemma, pred_pos = get_lemma_pos(node_prop)
        assigned_node_key = assign_lexical_node_key(node,node_prop,pred_lemma,pred_pos,dmrs_nxDG,underspecLemma,underspecProps,usage)
    # for structural predicates
    elif node_prop['instance'][0] in structural_pred or\
         dmrs_nxDG.nodes[node]['instance'] == 'pron' and dmrs_nxDG.nodes[node].get('pt')=="ZERO"\
         and dmrs_nxDG.nodes[node].get('pers') != 1:
        if underspecProps:
            assigned_node_key = node_prop['instance']
        else:
            default_node_key_props = list((key, node_prop[key]) for key in node_prop if not key=='lnk')
            assigned_node_key = frozenset(default_node_key_props)
    # for all other predicates
    else:
        if underspecProps:
            assigned_node_key = node_prop['instance']
        else:
            default_node_key_props = list((key, node_prop[key]) for key in node_prop if not key=='lnk')
            assigned_node_key = frozenset(default_node_key_props)
    return assigned_node_key
    
    
def assign_lexical_node_key(node,node_prop,pred_lemma,pred_pos,dmrs_nxDG,underspecLemma = False,underspecProps = False,usage = 'surface'):
    # if adj/adv
    if pred_pos == 'a' or unknown2pos.get(pred_pos) == 'a':
        assigned_node_key = assign_adjadv_node2key(node,node_prop,dmrs_nxDG,underspecLemma,underspecProps,usage)
    # if verb/verb-like
    elif pred_pos == 'v' or unknown2pos.get(pred_pos) == 'v':
        assigned_node_key = assign_verb_node2key(node,node_prop,pred_lemma,pred_pos,dmrs_nxDG,underspecLemma,underspecProps,usage) 
    # if noun/noun-like
    elif pred_pos == 'n' or unknown2pos.get(pred_pos) == 'n':
        assigned_node_key = assign_noun_node2key(node,node_prop,dmrs_nxDG,underspecLemma,underspecProps,usage)
    # if ellipsis_ref/ellipsis_expl (do,be,to)
    elif pred_lemma in ["ellipsis_expl","ellipsis_ref"]:
        assigned_node_key = assign_verb_node2key(node,node_prop,pred_lemma,pred_pos,dmrs_nxDG,underspecLemma,underspecProps,usage) 
    # if still not yet handled (pos apart from a,v,n)
    else:
        if underspecProps:
            if underspecLemma:
                assigned_node_key = "_[usp]_" + pred_pos
            else:
                assigned_node_key = node_prop['instance']
        else:
            default_node_key_props = list((key, node_prop[key]) for key in node_prop if not key=='lnk')
            assigned_node_key = frozenset(default_node_key_props)
    return assigned_node_key

def assign_adjadv_node2key(node,node_prop,dmrs_nxDG,underspecLemma = False,underspecProps = False, usage="surface"):
    if underspecProps:
        if underspecLemma:
            return "_[usp]_a"
        else:
            return node_prop['instance']
    adjadv_form_key = None
    # consider cvcarsort of arg1 of current node to determine adj vs adv
    out_edges = dmrs_nxDG.out_edges(nbunch=[node],data=True)
    arg1_cvarsort = 'abs' # ~1800 cases
    comp_superl_abs = 'abs'
    for out_edge in out_edges:
        edge_label = out_edge[2]['label']
        if edge_label.split("-")[0] == "ARG1":
            arg1_cvarsort = dmrs_nxDG.nodes[out_edge[1]]['cvarsort']
            break
    # consider if there is any incoming comp/su of arg1 of current node
    in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
    for in_edge in in_edges:
        edge_label = in_edge[2]['label']
        edge_source_pred = dmrs_nxDG.nodes[in_edge[0]]['instance']
        if edge_source_pred in ['comp','superl'] and edge_label.split("-")[0] == "ARG1":
            comp_superl_abs = edge_source_pred
            break
    
    if underspecLemma:
        adjadv_form_key = [("instance", "_[usp]_a"), ("ARG1-*",arg1_cvarsort,), ("comp/superl",comp_superl_abs,)]
    else:
        adjadv_form_key = [("instance", node_prop['instance']), ("ARG1-*",arg1_cvarsort,), ("comp/superl",comp_superl_abs,)]
        
    if usage == "order":
        if node_prop.get("tense") == None:
            tense = "abs"
        elif node_prop.get("tense") == "UNTENSED":
            tense = "UNTENSED"
        else:
            tense = "TENSED" 
        adjadv_form_key.append(("tense",tense))
        if node_prop.get("sf") == None:
            sf = "abs"
        else:
            sf = node_prop.get("sf")
        adjadv_form_key.append(("sf",sf))
        
    return frozenset(adjadv_form_key)

def assign_verb_node2key(node,node_prop,pred_lemma,pred_pos,dmrs_nxDG,underspecLemma = False,underspecProps = False, usage="surface"):
    if underspecProps:
        if underspecLemma:
            return "_[usp]_v"
        else:
            return node_prop['instance']
    elif underspecLemma:
        pred_lemma = "[usp]"
    verb_form_key = None
    # if verb <- parg_d, then past_part
    in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
    for in_edge in in_edges:
        edge_label = in_edge[2]['label']
        if dmrs_nxDG.nodes[in_edge[0]]['instance'] == 'parg_d':
            verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "past_part")]
            is_passive = True
            break
    # else if prog+, then pres_part
    if not verb_form_key and node_prop.get('prog') == '+':
        verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "pres_part")]
    # else if perf+, then past_part
    if not verb_form_key and node_prop.get('perf') == '+':
        verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "past_part")]
    # else if neg -> verb, base_form (except 'be') (should try neg ->* verb (have path to))
    if not verb_form_key and pred_lemma not in ['be','ellipsis_ref','ellipsis_expl']:
        in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
        for in_edge in in_edges:
            inedge_source_pred = dmrs_nxDG.nodes[in_edge[0]]['instance']
            edge_label = in_edge[2]['label']
            if inedge_source_pred == 'neg' and edge_label.split("-")[0] == "ARG1":
                verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "base_form/to_el")]
                is_neg = True
                break
    # else(if not passive nor negated), if verb(pres,-,-) -arg1-> 3rd person (rough arg1~=subj)
    ## not true in some _v case, e.g. expletive it not arg1 of verb)
    if not verb_form_key and (node_prop.get('tense'),
                                node_prop.get('prog'),
                                node_prop.get('perf')) == ('PRES','-','-'):
        out_edges = dmrs_nxDG.out_edges(nbunch=[node],data=True)
        # check if target of any outgoing ARG1-* edge is 'x'
        for out_edge in out_edges:
            edge_label = out_edge[2]['label']
            if edge_label.split("-")[0] == "ARG1":
                verb_subj = dmrs_nxDG.nodes[out_edge[1]]
                # make sure is nouny, but not e.g. _may_v_modal -arg1-> _hire_v_1
                if verb_subj['cvarsort'] == 'x' and verb_subj.get('pers') == 3 and verb_subj.get('num') == 'SG':
                    verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "third_per_sg")]
                    break
                # two additional case for 'be' -> 'are'/'am'
                elif pred_lemma in ['be','ellipsis_ref','ellipsis_expl'] and verb_subj['cvarsort'] == 'x' and\
                     (verb_subj.get('num') == 'PL' or verb_subj.get('pers') == 2):
                    verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "_are/do_el")]
                    break
                elif pred_lemma in ['be','ellipsis_ref','ellipsis_expl'] and verb_subj['cvarsort'] == 'x' and\
                     verb_subj.get('pers') == 1 and verb_subj.get('num') == 'SG':
                    verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "_am/do_el")]
                    break
    # else(if not passive, negated nor present tense), if past tense/subjunctiuve
    if not verb_form_key and node_prop.get('tense') == 'PAST':
        # handle was,were
        if pred_lemma in ['be','ellipsis_ref','ellipsis_expl']:
            # check if is subjunctive mood first
            if node_prop.get('mood') == 'SUBJUNCTIVE':
                verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "_were/did_el")]
            else:
                out_edges = dmrs_nxDG.out_edges(nbunch=[node],data=True)
                # check if target of any outgoing ARG1-* edge is 'x'
                for out_edge in out_edges:
                    edge_label = out_edge[2]['label']
                    if edge_label.split("-")[0] == "ARG1":
                        verb_subj = dmrs_nxDG.nodes[out_edge[1]]
                        # make sure is nominal, but not e.g. _may_v_modal -arg1-> _hire_v_1
                        if verb_subj['cvarsort'] == 'x' and verb_subj.get('pers') == 2 or\
                           verb_subj.get('num') == 'PL':
                            verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "_were/did_el")]
                        elif verb_subj['cvarsort'] == 'x' and\
                             (verb_subj.get('pers') == 1 or\
                             verb_subj.get('pers') == 3):
                            verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "_was/did_el")]
        else:
            verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "bare_past")]  
    if not verb_form_key:
        verb_form_key = [("lemma", pred_lemma),("pos", pred_pos),("verb_form", "base_form/to_el")]
#     if usage == 'order':
    if node_prop.get("sf") == None:
        sf = "abs"
    else:
        sf = node_prop.get("sf")
    verb_form_key.append(("sf",sf))
    return frozenset(verb_form_key)

def assign_noun_node2key(node,node_prop,dmrs_nxDG,underspecLemma = False,underspecProps = False,usage = 'surface'):
    if underspecProps:
        if underspecLemma:
            return "_[usp]_n"
        else:
            return node_prop['instance']
    if usage == 'order':
        if underspecLemma:
            noun_form_key =  [("instance", "_[usp]_n")] #, ("ind", ind) ]
        else:
            noun_form_key =  [("instance", node_prop['instance'])] #, ("ind", ind) ]
    elif usage == 'surface':
        num = 'abs'
    #    ind = 'abs'
        if 'num' in node_prop:
            num = node_prop['num']
    #    if 'ind' in node_prop:
    #        ind = node_prop['ind']
        if underspecLemma:
            noun_form_key =  [("instance", "_[usp]_n"), ("num",num,)] #, ("ind", ind) ]
        else:
            noun_form_key =  [("instance", node_prop['instance']), ("num",num,)] #, ("ind", ind) ]
    return frozenset(noun_form_key)




def predict_adjadv_node2surface(assigned_node_key,pred_lemma,node2surface2cnt):
    predict_surface = None
    # predict surface if new key in dict, else fallbacks
    if assigned_node_key in node2surface2cnt:
        predict_surface = node2surface2cnt[assigned_node_key].most_common()[0][0]
        # print (predict_surface, real_surface)
    # fallbacks
    else:
        arg1_cvarsort = comp_superl_abs = instance = None
        for prop,value in assigned_node_key:
            if prop == 'ARG1-*':
                arg1_cvarsort = value
            if prop == 'comp/superl':
                comp_superl_abs = value
            if prop == 'instance':
                instance = value
        
        xe_fallback_node_key = frozenset([("instance", instance), ("ARG1-*",arg1_cvarsort,), ("comp/superl",'abs',)])
        final_fallback_node_key = frozenset([("instance", instance), ("ARG1-*",'abs',), ("comp/superl",'abs',)])
                
        # adj case (unknown case of i)
        if arg1_cvarsort in ['x','i']:
            if xe_fallback_node_key in node2surface2cnt:
                predict_surface = node2surface2cnt[xe_fallback_node_key].most_common()[0][0]
            elif final_fallback_node_key in node2surface2cnt:
                predict_surface = node2surface2cnt[final_fallback_node_key].most_common()[0][0]                   
            else:
                predict_surface = pred_lemma
            # handle comp/superl
            if comp_superl_abs == 'comp':
                predict_surface = predict_surface + "er"
            elif comp_superl_abs == 'superl':
                predict_surface = predict_surface + "est"
        # adv case
        elif arg1_cvarsort == 'e':
            if xe_fallback_node_key in node2surface2cnt:
                predict_surface = node2surface2cnt[xe_fallback_node_key].most_common()[0][0]
            elif final_fallback_node_key in node2surface2cnt:
                predict_surface = node2surface2cnt[final_fallback_node_key].most_common()[0][0]                       
            else:
                predict_surface = pred_lemma + "ly"
            # don't handle comp/superl (assume most of them exist as more/most <lemma>-ly)
        else:
            if final_fallback_node_key in node2surface2cnt:
                predict_surface = node2surface2cnt[final_fallback_node_key].most_common()[0][0]                       
            else:
                predict_surface = pred_lemma
    return predict_surface

def predict_verb_node2surface(assigned_node_key,pred_lemma,node2surface2cnt):
    predict_surface = None
    #if assgined key in dict, else fallbacks
    if assigned_node_key in node2surface2cnt:
        predict_surface = node2surface2cnt[assigned_node_key].most_common()[0][0]
    # if unfound, fallback according to verb_form
    else:
        verb_form2suffix = {"bare_past":"ed","pres_part":"ing","past_part":"ed","third_per_sg":"s","base_form/to_el":""}
        
        verb_form = None
        for prop,value in assigned_node_key:                  
            if prop == 'verb_form':
                verb_form = value
        
        if verb_form in verb_form2suffix:
            # handle verb ending with e
            if pred_lemma[-1] == 'e' and verb_form in ['bare_past','pres_part','past_part']:
                predict_surface = pred_lemma[:-1] + verb_form2suffix[verb_form]
            else:
                predict_surface = pred_lemma + verb_form2suffix[verb_form]
        else:
            if pred_lemma == 'be':
                if verb_form == '_was/did_el':
                    predict_surface = 'was'
                elif verb_form == '_were/did_el':
                    predict_surface = 'were'
                elif verb_form == '_are/do_el':
                    predict_surface = 'are'
                elif verb_form == '_am/do_el':
                    predict_surface = 'am'
                else:
                    print (assigned_node_key)
            else:
                predict_surface = pred_lemma
    return predict_surface

def predict_noun_node2surface(assigned_node_key,pred_lemma,node2surface2cnt,VOWELS):
    predict_surface = None
    # if assigned key in dict, else fallbacks
    if assigned_node_key in node2surface2cnt:
        predict_surface = node2surface2cnt[assigned_node_key].most_common()[0][0]
    # if unfound, fallback to lemma + (s) if plural
    else:
        num = instance = None
        for prop,value in assigned_node_key:                  
            if prop == 'num':
                num = value
            if prop == 'instance':
                instance = value
                
        if num == 'SG':
            abs_fallback_node_key = frozenset([("instance", instance), ("num",'abs',)])
            if abs_fallback_node_key in node2surface2cnt:
                predict_surface = node2surface2cnt[abs_fallback_node_key].most_common()[0][0]
        elif num == 'abs':
            sg_fallback_node_key = frozenset([("instance", instance), ("num",'sg',)])
            if sg_fallback_node_key in node2surface2cnt:
                predict_surface = node2surface2cnt[sg_fallback_node_key].most_common()[0][0]
        if not predict_surface:
            predict_surface = pred_lemma
            if num == 'PL':
                if pred_lemma.endswith(('s', 'sh', 'h', 'x', 'z', 'o')):
                    predict_surface = predict_surface + 'es'
                elif pred_lemma.endswith(('f')):
                    predict_surface = predict_surface[0:-1] + 'ves'
                elif pred_lemma.endswith(('fe')):
                    predict_surface = predict_surface[0:-2] + 'ves'
                elif pred_lemma.endswith('y') and not pred_lemma[-2] in VOWELS:
                    predict_surface = predict_surface[0:-1] + 'ies'
                elif pred_lemma.endswith('y') and not pred_lemma[-2] in VOWELS:
                    predict_surface = predict_surface[0:-1] + 'ies'
                else:
                    predict_surface = predict_surface + 's'
    return predict_surface


# for rule extraction
def modify_surface(surface,node,dmrs_nxDG):
    mod_surface = surface
    in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
    for in_edge in in_edges:
        edge_source_pred = dmrs_nxDG.nodes[in_edge[0]]['instance']
        edge_label = in_edge[2]['label']
        if edge_source_pred[0] == '_' and edge_source_pred.split("_")[1][-1] == "-" and\
           edge_label.split("-")[0] == 'ARG1':
            pred_prefix = edge_source_pred.split("_")[1][0:-1]
            if surface.startswith(pred_prefix):
                mod_surface = surface[len(pred_prefix):]
                if surface[0] == '-':
                    mod_surface = surface[1:]
    return mod_surface

# postprocessing for surface prediction
def recover_surface(node,dmrs_nxDG,predict_surface,PREFIXES):
    in_edges = dmrs_nxDG.in_edges(nbunch=[node],data=True)
    for in_edge in in_edges:
        if 'subgraph_canon' in dmrs_nxDG.nodes[in_edge[0]]:
            continue
        edge_source_pred = dmrs_nxDG.nodes[in_edge[0]]['instance']
        edge_label = in_edge[2]['label']
        if edge_source_pred[0] == "_" and edge_source_pred.split("_")[1][-1] == "-" and edge_label.split("-")[0] == 'ARG1':
            if edge_source_pred.split("_")[1] in PREFIXES:
                predict_surface = edge_source_pred.split("_")[1][0:-1] + predict_surface
            else:
                predict_surface = edge_source_pred.split("_")[1] + predict_surface
    return predict_surface
    
    
if  __name__ =='__main__':
    pass
