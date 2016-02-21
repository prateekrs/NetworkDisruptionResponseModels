import networkx
import pandas
import pyomo
import pyomo.opt
import pyomo.environ as pe
import scipy
import itertools
import logging

def create_default_SuperNodeCSV(superGraph,rows=7,columns=7):
    print rows, columns    
    g = networkx.grid_2d_graph(rows,columns)
    # super_nodes_nodes_data.csv
    nodes = sorted(g.nodes())
    demands = scipy.ones(len(nodes))
    facility_allowed = scipy.ones(len(nodes))
    super_nodes_nodes_data = pandas.DataFrame()
    for u in superGraph.nodes():
        data = {'SuperNode':u, 'Node':nodes, 'CustDemand':demands, 'FacilityOK':facility_allowed}
        super_nodes_nodes_data = pandas.concat([super_nodes_nodes_data, pandas.DataFrame(data)])
    super_nodes_nodes_data.to_csv('super_nodes_nodes_data.csv', index=False)
    # super_arcs_data.csv
    dist = networkx.all_pairs_shortest_path_length(g)
    pairs = sorted(list(itertools.product(nodes,nodes)))
    distance = []
    for n1, n2 in pairs:
        distance.append( dist[n1][n2] )
    super_nodes_arcs_data = pandas.DataFrame()
    start, end = zip(*pairs)
    for u in superGraph.nodes():
        data = {'SuperNode':u, 'StartNode':start, 'EndNode':end, 'Distance':distance}
        super_nodes_arcs_data = pandas.concat([super_nodes_arcs_data, pandas.DataFrame(data)])
    super_nodes_arcs_data.to_csv('super_nodes_arcs_data.csv', index=False)


class SuperEdge:
    def __init__(self, p, snA, snB):
        self.p = p
        self.start = snA
        self.end = snB
        # Create the constraints for calculating reconfig cost
        self.model = model = pe.ConcreteModel()
        model.snA = snA
        model.snB = snB

        model.W = pe.Var( snA.model.fac_set*snA.model.fac_loc_set*snB.model.fac_loc_set , domain=pe.Binary)

        def move_rule(model, u, i, j):
            return model.W[u,i,j] >= model.snB.model.Zin[u,j] + model.snA.model.Zout[u,i] - 1

        model.MoveConst = pe.Constraint( snA.model.fac_set*snA.model.fac_loc_set*snB.model.fac_loc_set, rule=move_rule)


        def reconf_expr_rule(model):
            return self.p*sum( model.W[u,i,j] * snA.SN_arcs_data.ix[(i,j), 'Distance'] * snA.facility_data.ix[u,'UnitMovementCost'] for u in snA.fac_set for i in snA.fac_loc_set for j in snB.fac_loc_set )

        self.model.reconf_expr = reconf_expr_rule

class SuperNode:
    def __init__(self, snid, epsilon = .1, super_nodes_nodes_data = 'super_nodes_nodes_data.csv', 
            super_nodes_arcs_data = 'super_nodes_arcs_data.csv', 
            facility_data = 'facility_data.csv'):
        self.epsilon = epsilon
        # Load and filter data for this super node only
        self.SN_node_data = pandas.read_csv(super_nodes_nodes_data)
        self.SN_node_data = self.SN_node_data[ self.SN_node_data['SuperNode'] == snid ]
        self.SN_arcs_data = pandas.read_csv(super_nodes_arcs_data)
        self.SN_arcs_data = self.SN_arcs_data[ self.SN_arcs_data['SuperNode'] == snid ]
        self.facility_data = pandas.read_csv(facility_data)
        # Now create the sets 
        self.SN_node_data.set_index(['Node'], inplace=True)
        self.SN_node_data.sort_index(inplace=True)
        self.SN_arcs_data.set_index(['StartNode','EndNode'], inplace=True)
        self.SN_arcs_data.sort_index(inplace=True)
        self.facility_data.set_index(['Facility'], inplace=True)
        self.facility_data.sort_index(inplace=True)
        # Can df.reset_index() to go back
    
        self.cust_set = self.SN_node_data.index.unique()
        self.fac_loc_set = self.SN_node_data.index.unique()
        self.fac_set = self.facility_data.index.unique()

        self.createModel()

    def createModel(self):
        # Create the Pyomo Model
        self.model = model = pe.ConcreteModel()

        # Add the sets
        model.cust_set = pe.Set( initialize=self.cust_set )
        model.fac_loc_set = pe.Set( initialize=self.fac_loc_set)
        model.fac_set = pe.Set( initialize=self.fac_set )

        # Create the variables
        # Facility u at node j?
        model.Y = pe.Var(model.fac_set*model.fac_loc_set, domain=pe.Binary) 
        # Amount cust i gets from location j
        model.X = pe.Var(model.cust_set*model.fac_loc_set, domain=pe.NonNegativeReals)
        # Facility u at node j?
        model.Zin = pe.Var(model.fac_set*model.fac_loc_set, domain=pe.Binary) 
        # Facility u at node j?
        model.Zout = pe.Var(model.fac_set*model.fac_loc_set, domain=pe.Binary) 

        # Create the objective
        def obj_rule(model):
            return  sum( model.X[c,f] * self.SN_arcs_data.ix[(c,f),'Distance'] for c in self.cust_set for f in self.fac_loc_set)
        model.OBJ = pe.Objective(rule=obj_rule, sense=pe.minimize)

        # Create the constraints
        # Every customer is served
        def cust_serve_rule(model, c):
            if not self.SN_node_data.ix[c,'FacilityOK']:
                return pe.Constraint.Skip
            return sum( model.X[c,f] for f in self.fac_loc_set ) >= self.SN_node_data.ix[c,'CustDemand'] 
        model.CustServed = pe.Constraint(model.cust_set, rule=cust_serve_rule)

        # Can't serve more than capacity
        def cap_rule(model,f):
            return sum(model.X[c,f] for c in self.cust_set) <= sum( model.Y[u,f]*self.facility_data.ix[u,'Capacity'] for u in self.fac_set ) 
        model.CapRule = pe.Constraint(model.fac_loc_set, rule=cap_rule)

        # Can't place facilities where not allowed in this super node
        def fac_cutoff_rule(model,u,f):
            return model.Y[u,f] <= self.SN_node_data.ix[f,'FacilityOK']
        model.FacCutoff = pe.Constraint(model.fac_set*model.fac_loc_set, rule=fac_cutoff_rule)

        # Each facility on one location
        def fac_one_loc_rule(model,u):
            return sum(model.Y[u,f] for f in self.fac_loc_set) == 1 
        model.FacOneLoc = pe.Constraint(model.fac_set, rule=fac_one_loc_rule)
        
        # Each location one facility
        def loc_one_fac_rule(model,f):
            return sum(model.Y[u,f] for u in self.fac_set) <= 1 
        model.LocOneFac = pe.Constraint(model.fac_loc_set, rule=loc_one_fac_rule)

        # Z connection constraints
        def zin_rule(model,u,f):
            return model.Zin[u,f] == model.Y[u,f] 
        model.ZinConst = pe.Constraint(model.fac_set*model.fac_loc_set, rule=zin_rule)
        
        def zout_rule(model,u,f):
            return model.Zout[u,f] == model.Y[u,f] 
        model.ZoutConst = pe.Constraint(model.fac_set*model.fac_loc_set, rule=zout_rule)


        solver = pyomo.opt.SolverFactory('cplex')
        # Solve to get O*
        self.model.preprocess()
        results = solver.solve(self.model, tee=True, keepfiles=False, options_string="mip_tolerances_integrality=1e-9 mip_tolerances_mipgap=0")
        # Check that we actually computed an optimal solution
        if (results.solver.status != pyomo.opt.SolverStatus.ok):
            logging.warning('Check solver not ok?')
        if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
            logging.warning('Check solver optimality?')

        self.optObj = self.model.OBJ()
        # Now create the efficiency constraint and deactivate the objective
        self.model.OBJ.deactivate()

        def efficiency_rule(model):
            return model.OBJ.rule(model) <= (1+self.epsilon)*self.optObj
        model.EffConst = pe.Constraint(rule=efficiency_rule)


    def visualize(self):
        import matplotlib
        import pylab
        fig = pylab.gcf()
        ax = pylab.gca()

        for f in self.fac_loc_set:
            fac_type = None
            for u in self.fac_set:
                if self.model.Y[u,f].value > .99:
                    fac_type = self.facility_data.ix[u,'FacilityType']
                    num = u
            x,y = f.split(',')
            x = int(x[1:])
            y = int(y[:-1])
            if not fac_type:
                if self.SN_node_data.ix[f, 'FacilityOK']:
                    color = 'none'
                else:
                    color = 'gray'
                ax.add_patch( matplotlib.patches.Circle((x,y), .1, fc=color))
            elif fac_type == 'Big':
                ax.add_patch( matplotlib.patches.Circle((x,y), .4, fc='black', alpha=1))
                ax.text(x,y, str(num), horizontalalignment='center', verticalalignment='center')
            elif fac_type == 'Med':
                ax.add_patch( matplotlib.patches.Circle((x,y), .25, fc='red', alpha=.8))
                ax.text(x,y, str(num), horizontalalignment='center', verticalalignment='center')
            elif fac_type == 'Small':
                ax.add_patch( matplotlib.patches.Circle((x,y), .15, fc='green', alpha=.3))
                ax.text(x,y, str(num), horizontalalignment='center', verticalalignment='center')

        ax.relim()
        ax.autoscale_view(True,True,True)
        fig.canvas.draw()


def example1():
    # Create a super-graph
    sg = networkx.DiGraph()
    sg.add_edge('A','B')
    sg.add_edge('B','A')
    # Edit the problem data
    create_default_SuperNodeCSV(sg)

    df = pandas.read_csv('super_nodes_nodes_data.csv')
    df.ix[ (df.SuperNode == 'A') & (df.Node.str.extract('\(([0-9]*),').astype(int) <= 3), 'FacilityOK'] = 0
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('\(([0-9]*),').astype(int) >= 3), 'FacilityOK'] = 0
    df.to_csv('super_nodes_nodes_data.csv',index=False)
    return sg,'ex1'

def example2():
    sg = networkx.DiGraph()
    sg.add_edge('A','B')
    sg.add_edge('B','C')
    sg.add_edge('C','A')
    # Edit the problem data
    create_default_SuperNodeCSV(sg)
    df = pandas.read_csv('super_nodes_nodes_data.csv')
    df.ix[ (df.SuperNode == 'A') & (df.Node.str.extract('\(([0-9]*),').astype(int) <= 2), 'FacilityOK'] = 0
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('\(([0-9]*),').astype(int) +df.Node.str.extract('.*, ([0-9]*)\)').astype(int)   <= 5), 'FacilityOK'] = 0
    df.ix[ (df.SuperNode == 'C') & (df.Node.str.extract('.*, ([0-9]*)\)').astype(int)   <= 2), 'FacilityOK'] = 0
    df.to_csv('super_nodes_nodes_data.csv',index=False) 
    return sg,'ex2'

def example3(): 
# parameters and data required- supernode graph structure(edges and transition probabilities)
# grid size of supernode (rows and columns)
# inactive nodes in each supernode
# name of example
# name of facility data file
    sg = networkx.DiGraph()
    sg.add_edge('A','B')
    sg.add_edge('A','C')
    sg['A']['B']['probability']=0.75
    sg['A']['C']['probability']=0.25
    
    # Edit the problem data
    rows=4
    columns=4    
    create_default_SuperNodeCSV(sg,rows,columns)
    df = pandas.read_csv('super_nodes_nodes_data.csv')
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('\(([0-9]*),').astype(int) <= 1), 'FacilityOK'] = 0
    df.ix[ (df.SuperNode == 'C') & (df.Node.str.extract('\(([0-9]*),').astype(int) >= 2), 'FacilityOK'] = 0
    df.to_csv('super_nodes_nodes_data.csv',index=False) 
    return sg,'ex3','facility_data_example3.csv',epsilon

def example3_stoch(): 
# parameters and data required- supernode graph structure(edges and transition probabilities)
# grid size of supernode (rows and columns)
# inactive nodes in each supernode
# name of example
# name of facility data file
    sg = networkx.DiGraph()
    sg.add_edge('A','B')
    sg.add_edge('A','C')
    sg['A']['B']['probability']=0.5
    sg['A']['C']['probability']=0.5
    
    # Edit the problem data
    epsilon=5  
    rows=4
    columns=4    
    create_default_SuperNodeCSV(sg,rows,columns)
    df = pandas.read_csv('super_nodes_nodes_data.csv')
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('\(([0-9]*),').astype(int) <= 1), 'FacilityOK'] = 0
    df.ix[ (df.SuperNode == 'C') & (df.Node.str.extract('\(([0-9]*),').astype(int) >= 2), 'FacilityOK'] = 0
    df.to_csv('super_nodes_nodes_data.csv',index=False) 
    return sg,'ex3_stoch','facility_data_example3_stoch.csv',epsilon


def example4(): 
# parameters and data required- supernode graph structure(edges and transition probabilities)
# grid size of supernode (rows and columns)
# inactive nodes in each supernode
# name of example
# name of facility data file
# epsilon    
    sg = networkx.DiGraph()
    sg.add_edge('A','B')
    sg.add_edge('A','C')
    sg['A']['B']['probability']=0.75
    sg['A']['C']['probability']=0.25
    
    # Edit the problem data
    rows=5
    columns=5    
    create_default_SuperNodeCSV(sg,rows,columns)
    df = pandas.read_csv('super_nodes_nodes_data.csv')
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('\(([0-9]*),').astype(int) <= 0), 'FacilityOK'] = 0
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('\(([0-9]*),').astype(int) >= columns-1), 'FacilityOK'] = 0
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('.*, ([0-9]*)\)').astype(int) <= 0), 'FacilityOK'] = 0
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('.*, ([0-9]*)\)').astype(int) >= rows-1), 'FacilityOK'] = 0    
    df.ix[ (df.SuperNode == 'C') & (df.Node.str.extract('\(([0-9]*),').astype(int) >= 3), 'FacilityOK'] = 0
    df.to_csv('super_nodes_nodes_data.csv',index=False) 
    epsilon=0.5
    return sg,'ex4','facility_data_example4.csv',epsilon

def example4_stoch(): 
# parameters and data required- supernode graph structure(edges and transition probabilities)
# grid size of supernode (rows and columns)
# inactive nodes in each supernode
# name of example
# name of facility data file
# epsilon    
    sg = networkx.DiGraph()
    sg.add_edge('A','B')
    sg.add_edge('A','C')
    sg['A']['B']['probability']=0.75
    sg['A']['C']['probability']=0.25
    
    # Edit the problem data
    epsilon=0
    rows=5
    columns=5    
    create_default_SuperNodeCSV(sg,rows,columns)
    df = pandas.read_csv('super_nodes_nodes_data.csv')
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('\(([0-9]*),').astype(int) <= 0), 'FacilityOK'] = 0
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('\(([0-9]*),').astype(int) >= columns-1), 'FacilityOK'] = 0
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('.*, ([0-9]*)\)').astype(int) <= 0), 'FacilityOK'] = 0
    df.ix[ (df.SuperNode == 'B') & (df.Node.str.extract('.*, ([0-9]*)\)').astype(int) >= rows-1), 'FacilityOK'] = 0    
    df.ix[ (df.SuperNode == 'C') & (df.Node.str.extract('\(([0-9]*),').astype(int) >= 3), 'FacilityOK'] = 0
    df.to_csv('super_nodes_nodes_data.csv',index=False) 
    epsilon=0.5
    return sg,'ex4_stoch','facility_data_example4_stoch.csv',epsilon




if __name__ == '__main__':
    sg,name,facility_data_file,epsilon= example4_stoch()
    print(sg.edges)    
    
    # Create all the super-node and super-edge models
    for n in sg.nodes():
        sg.node[n]['model'] = SuperNode(n,facility_data = facility_data_file,epsilon=epsilon)

    for e in sg.edges():
        sg.edge[e[0]][e[1]]['model'] = SuperEdge(sg.edge[e[0]][e[1]]['probability'], sg.node[e[0]]['model'], sg.node[e[1]]['model'])

    # Create the reconfigurable model and solve
    reconf = pe.ConcreteModel()
    obj_expr = 0
    for n in sg.nodes():
        nmodel = sg.node[n]['model'].model
        obj_expr = obj_expr + nmodel.OBJ.rule( nmodel )
        setattr(reconf, 'sn_%s'%n, nmodel)

    for e in sg.edges():
        emodel = sg.edge[e[0]][e[1]]['model'].model
        obj_expr = obj_expr + emodel.reconf_expr( emodel )
        setattr(reconf, 'se_%s_%s'%(str(e[0]), str(e[1])), emodel)

    reconf.OBJ = pe.Objective(expr = obj_expr, sense=pe.minimize)

    reconf.preprocess()
    solver = pyomo.opt.SolverFactory('cplex')
    results = solver.solve(reconf, tee=True, keepfiles=False, options_string="mip_tolerances_integrality=1e-9 mip_tolerances_mipgap=0 mip_tolerances_absmipgap=1e-9")

    # Check that we actually computed an optimal solution
    if (results.solver.status != pyomo.opt.SolverStatus.ok):
        logging.warning('Check solver not ok?')
    if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
        logging.warning('Check solver optimality?')


    # Create the figures
    import pylab
    for n in sg.nodes():
	    pylab.clf()
	    sg.node[n]['model'].visualize()
	    pylab.savefig('%s-sn%s.png'%(name,n),dpi=300)


