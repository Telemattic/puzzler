import z3

def name_edge(tabA, tabB):
    return f"edge_{tabA}_{tabB}".replace(':','_')

class MosaicSat:

    def __init__(self, G):
        
        self.opt = opt = z3.Optimize()
        
        G = G.to_undirected()

        # the "good" nodes before we add the unknown node 'None'
        nodes = list(G.nodes)
        
        for n in nodes:
            G.add_edge(n, 'None')
            
        self.edges = edges = {}
        
        for a, b in G.edges:
            edges[b,a] = edges[a,b] = z3.Bool(name_edge(a,b))

        # exactly one of the edges is True
        for n in G.nodes:
            if n != 'None':
                opt.add(z3.PbEq(tuple((edges[ab], 1) for ab in G.edges(n)), 1))

        # minimize the number of tabs resolved as Unknown
        v = [z3.If(edges[n,'None'],1,0) for n in nodes]
        self.objective = opt.minimize(z3.Sum(v))

    def reject(self, feature_pairs):

        edges = self.edges
        v = [edges[str(a),str(b)] for a,b in feature_pairs]
        self.opt.add(z3.Not(z3.And(v)))

    def check(self):

        opt = self.opt
        edges = self.edges

        if opt.check() != z3.sat:
            print("No solution. :<")
            return None

        retval = []
        model = opt.model()
        for (a, b), e in sorted(edges.items()):
            if a < b and model[e] and a != 'None' and b != 'None':
                # print(f"{a}={b}")
                retval.append((a,b))
                
        print(f"Unknown tabs: {self.objective.value()}")

        return retval

