import networkx as nx
import pandas as pd
import cplex

adjacency_list={1:[(2,7,30),(3,6,20)],2:[(3,5,25),(4,4,10)],3:[(4,2,20),(5,2,25)],4:[(5,1,20)],5:[]}
def create_graph():
    G=nx.DiGraph() 
    for i in adjacency_list:
        if i not in G.nodes():
            G.add_node(i)

        for j in adjacency_list[i]:
            G.add_edge(i,j[0],cost=j[1],ub=j[2])

    print G.nodes()
    print G.edges()

def create_network():
    G=nx.DiGraph()
    network_data=pd.read_csv("F:\\PhD Work\\Disruption network\\Example_network_2.csv")
    print network_data.columns

    nodes=list(set(list(network_data['Node1'])+list(network_data['Node2'])))
    node_demands={1:-1,2:0,3:0,4:0,5:0,6:0,7:1}
    for i in node_demands:
        G.add_node(i,demand=node_demands[i])
    
    for i in network_data.index:
        node1=network_data['Node1'][i]
        node2=network_data['Node2'][i]
        edge_cost=int(network_data['Cost'][i])
        edge_ub=int(network_data['Upperbound'][i])
        G.add_edge(node1,node2,weight=edge_cost,capacity=edge_ub)
        
    flowDict = nx.min_cost_flow(G)
#    print flowDict,nx.min_cost_flow_cost(G)
#    print nodes
#    print G.nodes()
#    print G.edges()
    return G

def min_cost_flow_optimization():
    G=create_network()
    print 
    model=cplex.Cplex()
    model.set_problem_name('Min Cost Flow Optimization')

    for e in G.edges():
        print e, G[e[0]][e[1]]['weight'],
        model.variables.add( obj=[G[e[0]][e[1]]['weight']], 
                lb=[0], 
                ub=[G[e[0]][e[1]]['capacity']], 
                types=model.variables.type.continuous,
                names=["x_"+str(e[0])+"_"+str(e[1])])       
    print '\n',G.node[1]['demand'],'\n',G.node[2]['demand']
    print G.predecessors(1),G.predecessors(2)
    
    for i in G.nodes():
        ind = []
        val = []
        for j in G.predecessors(i):
            ind.append( "x_"+str(j)+"_"+str(i))
            val.append( 1 )
        for j in G.successors(i):
            ind.append( "x_"+str(i)+"_"+str(j) )
            val.append( -1 )
        rhs = G.node[i]['demand']
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], 
                        senses='E', 
                        rhs=[rhs])

    model.set_problem_type( model.problem_type.LP )

    model.solve()

#    print 'Solution: ', model.solution.get_status_string()
#    print 'Objective: ', model.solution.get_objective_value()


def disruption_optimization():
    model=cplex.Cplex()
    model.set_problem_name('Disruption Optimization')
    
    disruptions={0:[],1:[4]}
    theta={}
    epsilon={}
    for epsilon_step in range(0,11):
        epsilon_step=float(epsilon_step/100.0)
        print epsilon_step
        for disrupt in disruptions:
            G=create_network()
            G.remove_nodes_from(disruptions[disrupt])
            if disrupt==0:
                original_edges=G.edges()
#                print "original_edges",original_edges
            removed_edges=list(set(original_edges)-set(G.edges()))
            theta[disrupt]=nx.min_cost_flow_cost(G)
            epsilon[disrupt]=epsilon_step*theta[disrupt]
#            print theta[disrupt],epsilon[disrupt]

            #adding variable z representing difference in absolute values of original and disruption flows
            if disrupt!=0:
                for e in original_edges:
                    model.variables.add( obj=[1], 
                            lb=[0], 
                            ub=[cplex.infinity], 
                            types=model.variables.type.continuous,
                            names=["z_0_"+str(disrupt)+"_"+str(e[0])+"_"+str(e[1])])

            #adding variables x representing flows on edges
            for e in G.edges():
#                print e, G[e[0]][e[1]]['weight'],
                model.variables.add( obj=[0], 
                        lb=[0], 
                        ub=[G[e[0]][e[1]]['capacity']], 
                        types=model.variables.type.continuous,
                        names=["x_"+str(disrupt)+"_"+str(e[0])+"_"+str(e[1])])
                
#            print '\n',G.node[1]['demand'],'\n',G.node[2]['demand']
#            print G.predecessors(1),G.predecessors(2)

            #adding flow balance constraints
            for i in G.nodes():
                ind = []
                val = []
                for j in G.predecessors(i):
                    ind.append( "x_"+str(disrupt)+"_"+str(j)+"_"+str(i))
                    val.append( 1 )
                for j in G.successors(i):
                    ind.append( "x_"+str(disrupt)+"_"+str(i)+"_"+str(j) )
                    val.append( -1 )
                rhs = G.node[i]['demand']
                model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], 
                                senses='E', 
                                rhs=[rhs])

            #adding epsilon-optimality constraints
            ind = []
            val = []
            rhs = theta[disrupt]+epsilon[disrupt]
            for e in G.edges():
                ind.append( "x_"+str(disrupt)+"_"+str(e[0])+"_"+str(e[1]))
                val.append( G[e[0]][e[1]]['weight'] )
            model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], 
                                senses='L', 
                                rhs=[rhs])    
            #adding absolute value constraints z=|x| -> z>=x and z>=-x
            if disrupt!=0:
                for e in G.edges():
                    ind=["z_0_"+str(disrupt)+"_"+str(e[0])+"_"+str(e[1]),"x_"+str(0)+"_"+str(e[0])+"_"+str(e[1]), "x_"+str(disrupt)+"_"+str(e[0])+"_"+str(e[1])]
                    val=[1,-1,1]
                    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], 
                                    senses='G', 
                                    rhs=[0])             
                    val=[1,1,-1]
                    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], 
                                    senses='G', 
                                    rhs=[0])

                for e in removed_edges:
                    ind=["z_0_"+str(disrupt)+"_"+str(e[0])+"_"+str(e[1]),"x_"+str(0)+"_"+str(e[0])+"_"+str(e[1])]
                    val=[1,-1]
                    rhs =0
                    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], 
                                    senses='G', 
                                    rhs=[0])                
                    val=[1,1]
                    rhs =0
                    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], 
                                    senses='G', 
                                    rhs=[0])
                    
            #adding dummy variables to keep track of costs for each disruption scenario
            model.variables.add( obj=[0], 
                        lb=[0], 
                        ub=[cplex.infinity], 
                        types=model.variables.type.continuous,
                        names=["cost_disruption_"+str(disrupt)])

            ind = ["cost_disruption_"+str(disrupt)]
            val = [1]
            rhs = 0
            for e in G.edges():
                ind.append( "x_"+str(disrupt)+"_"+str(e[0])+"_"+str(e[1]))
                val.append( -G[e[0]][e[1]]['weight'] )
            model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], 
                                senses='E', 
                                rhs=[rhs])                                                                                                   
        model.set_problem_type( model.problem_type.LP )

        model.solve()

        print 'Solution: ', model.solution.get_status_string()
        print 'Objective: ', model.solution.get_objective_value()

        for disrupt in disruptions:
            print  model.solution.get_values("cost_disruption_"+str(disrupt))
        
    
disruption_optimization()
