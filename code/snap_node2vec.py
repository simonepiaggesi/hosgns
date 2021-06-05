from gem.embedding.node2vec import node2vec
from subprocess import call
import tempfile
from gem.utils import graph_util
from time import time
import sys,os

HOME = os.path.expanduser("~")

def loadWalks(file_name):
    walks = []
    with open(file_name, 'r') as f:
        for line in f:
            walk = list(map(int,line.strip().split()))
            walks.append(walk)
    return walks

class snap_node2vec(node2vec):
    
    def sample_random_walks(self, graph=None, edge_f=None,
                        is_weighted=False, no_python=False, directed=False):
        args = [HOME+"/snap/examples/node2vec/node2vec"]
        if directed == False:
            graph_mod = graph.to_undirected()
        elif directed ==True:
            graph_mod = graph

        with tempfile.TemporaryDirectory(dir = './') as dname:
            original_graph = dname + '/node2vec_test.graph'
            emb_result = dname + '/node2vec_test.walks'
            graph_util.saveGraphToEdgeListTxtn2v(graph_mod, original_graph)
            args.append("-i:%s" % original_graph)
            args.append("-o:%s" % emb_result)
            args.append("-d:%d" % self._d)
            args.append("-l:%d" % self._walk_len)
            args.append("-r:%d" % self._num_walks)
            args.append("-k:%d" % self._con_size)
            args.append("-e:%d" % self._max_iter)
            args.append("-p:%f" % self._ret_p)
            args.append("-q:%f" % self._inout_p)
            args.append("-v")
            if directed ==True:
                args.append("-dr")
            args.append("-w")
            args.append("-ow")
            t1 = time()
            try:
                call(args)
            except Exception as e:
                print(str(e))
                raise Exception('node2vec not found. Please compile snap, place node2vec in the system path and grant executable permission')
            self._W = loadWalks(emb_result)
            t2 = time()
        return self._W, (t2 - t1)

