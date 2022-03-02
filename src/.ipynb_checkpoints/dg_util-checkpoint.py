import networkx as nx
# from pygraphviz import AGraph
import re
from pprint import pprint
from textwrap import wrap
from random import randrange
# from delphin.derivation import UDFTerminal, UDFNode

class Erg_DiGraphs:
    def __init__(self):
        self.snt = ""
        self.deriv_dg = nx.DiGraph()
        self.syn_tree_dg = nx.DiGraph()
        self.dmrs_dg = nx.MultiDiGraph()
        
    def mrp_json_to_directed_graph(mrp_json):
        pass
    
    @staticmethod
    def split_line(s):
        return "\n".join(wrap(s, 100))
    
    @staticmethod
    def ppprint(s):
        layer = 0
        for idx,c in enumerate(s):
            if c == '+':
                if s[idx+1] != ' ':
                    print()
                    for i in range(layer):
                        print (end = " ")
            if c != '\\':
                print (c, end ="")
            if c in ['(','[']:
                layer += 2
                print ()
                for i in range(layer):
                    print (end = " ")
            elif c in [')',']']:
                layer -= 2
                print ()
                for i in range(layer):
                    print (end = " ")
    
    @staticmethod
    def to_seq_ag(ag, top_node_id, snt, simplified=True):
        '''
        Add position attributes to ag graph.
        return an pygraphviz AGraph.
        '''
        #if ag_type == 'dmrs':
        #print (ag)
        #print ()
        try:
            for node_id in ag.iternodes():
                node = ag.get_node(node_id)
                #print (node.attr['label'])
                an = [(int(re.split('[( , )]',prop_str.strip())[1]),int(re.split('[( , )]',prop_str.strip())[2]))
                                    for prop_str in node.attr['label'].split("\n") if prop_str.startswith("(")]
                an_from, an_to = an[0][0], an[0][1]
                pos_y = 0.0
                if node.attr['label'].split("\n")[0][0] != '_':
                    pos_y = sum([ord(c) for c in node.attr['label'].split("\n")[0]])%47 * 11
                    print (pos_y)
            
                node.attr['pos']="%f,%f"%(float(an_from+an_to)*21,pos_y)
                if node_id == str(top_node_id):
                    node.attr['shape'] = "box"
                ag.edge_attr['fontsize'] = 8
                if simplified:
                    label_list = node.attr['label'].split("\n")
                    node.attr['label'] = label_list[0] + "\n" + label_list[-1]
        except Exception as e:
            print (e)
        return ag
    
    def init_snt(self,snt):
        self.snt = snt
        
    def init_syn_tree(self,syn_tree,draw = False):
        '''
        First convert syntactic tree to dictionary {'id':,'children':[{...}]},
        then convert to DiGraph and store
        '''
        id_in = 0
        syn_tree_dict = dict()
        def dfs_to_dict(par_node, syn_tree_node, edge_label = 'U'):
            nonlocal id_in
            par_node['id'] = id_in
            id_in += 1
            par_node['edge_label'] = edge_label
            if isinstance(syn_tree_node, str):
                par_node['label'] = syn_tree_node
            else:
                par_node['label'] = syn_tree_node[0]
                if len(syn_tree_node) > 1:
                    par_node['children'] = []
                    # binary rule
                    if len(syn_tree_node) == 3:
                        par_node['children'].append(dict())
                        dfs_to_dict(par_node['children'][0], syn_tree_node[1], edge_label = 'L')
                        par_node['children'].append(dict())
                        dfs_to_dict(par_node['children'][1], syn_tree_node[2], edge_label = 'R')
                    # unary rule
                    elif len(syn_tree_node) == 2:
                        par_node['children'].append(dict())
                        dfs_to_dict(par_node['children'][0], syn_tree_node[1], edge_label = 'U')

        dfs_to_dict(syn_tree_dict, syn_tree, edge_label = 'U')
        
        syn_dg_data = syn_tree_dict
        tmp_syn_tree_dg = nx.readwrite.json_graph.tree_graph(syn_dg_data)
        # add edge label L, R, U and remove from node attr
        for node, node_prop in tmp_syn_tree_dg.nodes(data = True):
            in_edge = list(tmp_syn_tree_dg.in_edges(nbunch = [node]))
            if not in_edge:
                continue
            par, _ = in_edge[0]
            tmp_syn_tree_dg.edges[(par, node)]['label'] = node_prop['edge_label']
            del tmp_syn_tree_dg.nodes[node]['edge_label']
        self.syn_tree_dg = tmp_syn_tree_dg
        self.syn_tree_dg.graph['root'] = 0
    
    def init_erg_deriv_from_nxDG(self, deriv_nxDG, draw = False):
        if draw:
            for node, node_prop in deriv_nxDG.nodes(data = True):
#                 print (node_prop)
                if 'entity' in deriv_nxDG.nodes[node]:
                    deriv_nxDG.nodes[node]['label'] = "\n".join([Erg_DiGraphs.split_line(str(deriv_nxDG.nodes[node]['entity'])), str(node)])
                elif 'form' in deriv_nxDG.nodes[node]:
                    deriv_nxDG.nodes[node]['label'] = "\n".join([deriv_nxDG.nodes[node]['form'], str(node)])
                for prop in node_prop:
                    if prop not in ['entity', 'form', 'label', 'anchor_from', 'anchor_to']:
                        deriv_nxDG.nodes[node]['label'] += "\n" + prop + ": " + Erg_DiGraphs.split_line(str(deriv_nxDG.nodes[node][prop]))
                if 'anchor_from' in deriv_nxDG.nodes[node]:
                    deriv_nxDG.nodes[node]['label'] += "\n" + "<" + deriv_nxDG.nodes[node]['anchor_from'] + ":" + deriv_nxDG.nodes[node]['anchor_to'] + ">" 
        self.deriv_dg = deriv_nxDG
        

    def init_erg_deriv(self, deriv, draw = False):
        NON_TERM_FIELDS = ['entity','score','start','end']
        TERM_FIELDS = ['form']
#         print (
#             getattr(deriv,'id'),
#             deriv.entity,
#             deriv.score,
#             deriv.start,
#             deriv.end,
#             deriv.daughters,
#     #         deriv.head,
#             deriv.type,
#         )
        
        def parse_anchor_from(tfs_str):
            x = re.search(r"\+FROM \#\d*=\\\"(\d+)\\\"", tfs_str)
            y = re.search(r"\+FROM \\\"(\d+)\\\"", tfs_str)
            if x != None:
                return x.group(1)
            elif y != None:
                return y.group(1)
            else:
                print ("err\n",tfs_str)
                return None
            
        def parse_anchor_to(tfs_str):
            x = re.search(r"\+TO \\\"(\d+)\\\"", tfs_str)
            if x != None:
                return x.group(1)
            else:
                print ("err\n",tfs_str)
                return None
        
        def add_node_edge(par_node, node, edge_label = 'U'):
            #print (node.to_dict())
            node_anchors = None
            node_id = None
            FIELDS = None
            if isinstance(node, UDFNode):
                #non-terminal nodes
                FIELDS = NON_TERM_FIELDS
                node_id = getattr(node,'id')
                
            elif isinstance (node, UDFTerminal):
                #terminal nodes
                FIELDS = TERM_FIELDS
                # more than one token
                if len(getattr(node,'tokens')) > 1:
                    # print (getattr(node,'form'))
                    pass
#                     for token in getattr(node,'tokens'):
#                         Erg_DiGraphs.ppprint (token.tfs)
#                         print ()
                node_id = getattr(node,'tokens')[0].id
                node_anchors = (str(min([int(parse_anchor_from(token.tfs))
                                     for token in getattr(node,'tokens')])),
                                str(max([int(parse_anchor_to(token.tfs))
                                     for token in getattr(node,'tokens')]))
                               )
            self.deriv_dg.add_node(node_id)
            if node_anchors != None:
                self.deriv_dg.nodes[node_id]['anchor_from'] = node_anchors[0]
                self.deriv_dg.nodes[node_id]['anchor_to'] = node_anchors[1]
                
            for FIELD in FIELDS:
                self.deriv_dg.nodes[node_id][FIELD] = str(getattr(node,FIELD))
                
            if par_node != None:
                self.deriv_dg.add_edge(par_node.id, node_id, label = edge_label)
            
            if draw:
                if 'entity' in self.deriv_dg.nodes[node_id]:
                    self.deriv_dg.nodes[node_id]['label'] = "\n".join([self.deriv_dg.nodes[node_id]['entity'], str(node_id)])
                elif 'form' in self.deriv_dg.nodes[node_id]:
                    self.deriv_dg.nodes[node_id]['label'] = "\n".join([self.deriv_dg.nodes[node_id]['form'], str(node_id)])
                if 'anchor_from' and 'anchor_to' in self.deriv_dg.nodes[node_id]:
                    self.deriv_dg.nodes[node_id]['label'] += "\n" + "<" + self.deriv_dg.nodes[node_id]['anchor_from'] + ":" + self.deriv_dg.nodes[node_id]['anchor_to'] + ">" 
                
        
        #add nodes
        def dfs(par_node, deriv_node, edge_label):
            add_node_edge(par_node, deriv_node, edge_label)
            if hasattr(deriv_node, 'daughters'):
                if len(deriv_node.daughters) == 2:
                    dfs(deriv_node, deriv_node.daughters[0], edge_label = 'L')
                    dfs(deriv_node, deriv_node.daughters[1], edge_label = 'R')
                elif len(deriv_node.daughters) == 1:
                    dfs(deriv_node, deriv_node.daughters[0], edge_label = 'U')
        dfs(None, deriv, edge_label = "U")
        self.deriv_dg.graph['root'] = getattr(deriv,'id')
    

    def init_dmrsjson(self, dmrsjson, draw = False):
        is_good_dmrs = True
        self.dmrs_dg.graph['index'] = dmrsjson.get('index')
        self.dmrs_dg.graph['top'] = dmrsjson['top']
        for node in dmrsjson['nodes']:
            self.dmrs_dg.add_node(node['nodeid'])
            self.dmrs_dg.nodes[node['nodeid']]['instance'] = node['predicate']
            try:
                if 'sortinfo' in node:
                    for prop in node['sortinfo']:
                        if prop == 'cvarsort':
                            self.dmrs_dg.nodes[node['nodeid']][prop.lower()] = node['sortinfo'][prop]
                        else:
                            self.dmrs_dg.nodes[node['nodeid']][prop.lower()] = node['sortinfo'][prop].upper()
                if 'carg' in node:
                    self.dmrs_dg.nodes[node['nodeid']]['carg'] = node['carg']
                self.dmrs_dg.nodes[node['nodeid']]['lnk'] = "<{}:{}>".format(str(node['lnk']['from']), str(node['lnk']['to']))
            except:
                print (self.dmrs_dg.nodes[node['nodeid']])
                is_good_dmrs = False
            if draw:
                props_key = list(self.dmrs_dg.nodes[node['nodeid']].keys())
                self.dmrs_dg.nodes[node['nodeid']]['label'] = "\n".join([prop_key + ": " + self.dmrs_dg.nodes[node['nodeid']][prop_key] for prop_key in props_key])
                
        for edge in dmrsjson['links']:
            self.dmrs_dg.add_edge(edge['from'],edge['to'], label=edge['rargname']+"/"+edge['post'])
        return is_good_dmrs
    
    def init_dmrs_from_nxDG(self,dmrs_nxDG, draw = False):
        if draw:
            for node, node_prop in dmrs_nxDG.nodes(data = True):
                props_key = list(dmrs_nxDG.nodes[node].keys())
                dmrs_nxDG.nodes[node]['label'] = "\n".join([prop_key + ": " + Erg_DiGraphs.split_line(str(dmrs_nxDG.nodes[node][prop_key])) for prop_key in props_key if props_key]) # + [Erg_DiGraphs.split_line(str(node))])
        self.dmrs_dg = dmrs_nxDG
            
if  __name__ =='__main__':
    pass