import networkx as nx
import itertools
import cplex

def create_network(disrupted_nodes):
    n=4
    node_list=[i+1 for i in range(n*n)]

    G=nx.Graph()
    for i in node_list:
        if i%n!=1:
#            print i-1,i
            G.add_edge(i-1,i,weight=1)
        if i%n!=0:
#            print i,i+1
            G.add_edge(i,i+1,weight=1)
        if (i/n!=0) and (i!=n):
#            print i-n,i
            G.add_edge(i-n,i,weight=1)
        if i/n<n-1:
#            print i,i+n
            G.add_edge(i,i+n,weight=1)   


#    print G.edges()

#    for i,j in G.edges():
#        print i,j, G.edge[i][j]['weight']
    '''
    scenario_edges=list(itertools.combinations(disrupted_nodes,2))
    for j in scenario_edges:
        if j in G.edges():
            x,y=j
            G.edge[x][y]['weight']=5
    '''
    length=nx.all_pairs_dijkstra_path_length(G)
    
    return (length,G)


def deterministic_facility_location(G,cost,facilities,demand,s,A):
    model=cplex.Cplex()
    model.set_problem_name('Deterministic Facility Location')
    for i in G.nodes():
        for j in G.nodes():
            model.variables.add( obj=[cost[i][j]], 
                    lb=[0], 
                    ub=[demand[j]], 
                    types=model.variables.type.integer,
                    names=["X_"+str(s)+"_"+str(i)+"_"+str(j)])

    for u in facilities:
        for i in G.nodes():
            model.variables.add( obj=[0], 
                    lb=[0], 
                    ub=[cplex.infinity], 
                    types=model.variables.type.binary,
                    names=["Y_"+str(s)+"_"+str(u)+"_"+str(i)])


    for i in G.nodes():
        for j in G.nodes():
            ind = []
            val = []
            rhs = 0                
            ind.append( "X_"+str(s)+"_"+str(i)+"_"+str(j))
            val.append( 1 )

            for u in facilities:
                ind.append( "Y_"+str(s)+"_"+str(u)+"_"+str(i))
                val.append( -demand[j] )

            model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='L', rhs=[rhs])        

        #constraint 10
        for i in G.nodes():
                for u in facilities:
                    ind = []
                    val = []
                    rhs=A[i]
                    ind.append( "Y_"+str(s)+"_"+str(u)+"_"+str(i))
                    val.append( 1 )
                    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='L', rhs=[rhs])        


    for i in G.nodes():
        ind = []
        val = []
        rhs = 0
        for j in G.nodes():
            ind.append( "X_"+str(s)+"_"+str(i)+"_"+str(j))
            val.append( 1 )
        for u in facilities:
                ind.append( "Y_"+str(s)+"_"+str(u)+"_"+str(i))
                val.append( -facilities[u]['Cap'] )
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='L', rhs=[rhs])


    for j in G.nodes():
        ind = []
        val = []
        rhs =  demand[j]
        for i in G.nodes():
            ind.append( "X_"+str(s)+"_"+str(i)+"_"+str(j))
            val.append( 1 )
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='G',rhs=[rhs])

    for u in facilities:
        ind=[]
        val=[]
        rhs=1
        for n in G.nodes():
            ind.append("Y_"+str(s)+"_"+str(u)+"_"+str(n))
            val.append(1)
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='L',rhs=[rhs])

    model.set_problem_type( model.problem_type.MILP )
    model.solve()

    print 'Solution: ', model.solution.get_status_string()
    return model.solution.get_objective_value()

    for u in facilities:
        for i in G.nodes():
            if model.solution.get_values("Y_"+str(s)+"_"+str(u)+"_"+str(i))==1:
                print "Y_"+str(s)+"_"+str(u)+"_"+str(i)

                
def facility_location():
    model=cplex.Cplex()
    model.set_problem_name('Facility Location')

    theta={}                 #Scenario 0 always has no nodes to be removed
    epsilon=0.5            #Specify epsilon values

#    scenarios={0:{},1:(1,2,5,6),2:(3,4,7,8),3:(9,10,13,14),4:(11,12,15,16)}
    scenarios={0:{},1:(1,2,5,6),2:(3,4,7,8)}
    edges={(0,1):1,(1,2):1}
    facilities={1:{'type':'H','Cap':10},2:{'type':'H','Cap':10},3:{'type':'L','Cap':10},4:{'type':'L','Cap':10}}
    reconfig_cost={'L':10, 'H':50}
    demand={}

    

    cost_orig,G=create_network(scenarios[0])
    A={}
    for j in G.nodes():
        demand[j]=1
    
        
        
    for s in scenarios:
#        cost,G=create_network(scenarios[s])
        cost,G=(cost_orig,G)
        
        for j in G.nodes():
            A[j]=1
            
        for j in scenarios[s]:
            A[j]=0
            

        z_opt=deterministic_facility_location(G,cost,facilities,demand,s,A)
#        raw_input(s)
        
        #Adding variables X_s_i_j and Y_s_u_i
        for i in G.nodes():
            for j in G.nodes():
                model.variables.add( obj=[0], 
                        lb=[0], 
                        ub=[demand[j]], 
                        types=model.variables.type.integer,
                        names=["X_"+str(s)+"_"+str(i)+"_"+str(j)])

        for u in facilities:
            for i in G.nodes():
                model.variables.add( obj=[0], 
                        lb=[0], 
                        ub=[cplex.infinity], 
                        types=model.variables.type.binary,
                        names=["Y_"+str(s)+"_"+str(u)+"_"+str(i)])
                        
        #constraint 2
        for i in G.nodes():
            for j in G.nodes():
                ind = []
                val = []
                rhs = 0                
                ind.append( "X_"+str(s)+"_"+str(i)+"_"+str(j))
                val.append( 1 )

                for u in facilities:
                    ind.append( "Y_"+str(s)+"_"+str(u)+"_"+str(i))
                    val.append( -demand[j] )

                model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='L', rhs=[rhs])        

       #constraint 3
        for i in G.nodes():
            ind = []
            val = []
            rhs = 0
            for j in G.nodes():
                ind.append( "X_"+str(s)+"_"+str(i)+"_"+str(j))
                val.append( 1 )
            for u in facilities:
                    ind.append( "Y_"+str(s)+"_"+str(u)+"_"+str(i))
                    val.append( -facilities[u]['Cap'] )
            model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='L', rhs=[rhs])

        #constraint 4
        for j in G.nodes():
            ind = []
            val = []
            rhs =  demand[j]
            for i in G.nodes():
                ind.append( "X_"+str(s)+"_"+str(i)+"_"+str(j))
                val.append( 1 )
            model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='G',rhs=[rhs])
 
        #Constraint 6
        for u in facilities:
            ind=[]
            val=[]
            rhs=1
            for n in G.nodes():
                ind.append("Y_"+str(s)+"_"+str(u)+"_"+str(n))
                val.append(1)
            model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='L',rhs=[rhs])


        #constraint 5
        for i in G.nodes():
                for u in facilities:
                    ind = []
                    val = []
                    rhs=A[i]
                    ind.append( "Y_"+str(s)+"_"+str(u)+"_"+str(i))
                    val.append( 1 )
                    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='L', rhs=[rhs])        

        #Adding variables Z_in_s_u_n and Z_out_s_u_n
        for u in facilities:
            for n in G.nodes():
                
#                if s==0:
                model.variables.add( obj=[0], 
                        lb=[0], 
                        ub=[cplex.infinity], 
                        types=model.variables.type.binary,
                        names=["Z_out_"+str(s)+"_"+str(u)+"_"+str(n)])
                '''
                model.variables.add( obj=[0], 
                            lb=[0], 
                            ub=[cplex.infinity], 
                            types=model.variables.type.binary,
                            names=["Z_in_"+str(s+1)+"_"+str(u)+"_"+str(n)])
                model.variables.add( obj=[0], 
                            lb=[0], 
                            ub=[cplex.infinity], 
                            types=model.variables.type.binary,
                            names=["Z_out_"+str(s+1)+"_"+str(u)+"_"+str(n)])               
                '''
        #Constraint 7
        for u in facilities:
                for n in G.nodes():
                    ind=[]
                    val=[]
                    rhs=0
                    ind.append("Z_out_"+str(s)+"_"+str(u)+"_"+str(n))
                    val.append(1)
                    ind.append("Y_"+str(s)+"_"+str(u)+"_"+str(n))
                    val.append(-1)
                    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='E',rhs=[rhs])


        '''
        #Constraint 8
        for i in G.nodes():
            for j in G.nodes():
                for u in facilities:
                    ind=[]
                    val=[]
                    rhs=0
                    ind.append("w_"+str(s)+"_"+str(u)+"_"+str(i)+"_"+str(j))
                    val.append(1)
                    ind.append("Z_out_"+str(s)+"_"+str(u)+"_"+str(j))
                    val.append(-1)
                    ind.append("Z_in_"+str(s)+"_"+str(u)+"_"+str(i))
                    val.append(1)
                    model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='G',rhs=[rhs])
        '''
        #constraint 9
        ind = []
        val = []
        rhs = (1+epsilon)*z_opt
        for i in G.nodes():
            for j in G.nodes():
                ind.append( "X_"+str(s)+"_"+str(i)+"_"+str(j))
                val.append( cost[i][j] )
        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='L',rhs=[rhs])        
                        
    
            #Adding w_s_u_ij
    for e in edges:            
            for u in facilities:
                for i in G.nodes():
                    for j in G.nodes():
                        (s1,s2)=e
                        model.variables.add( obj=[edges[e]*reconfig_cost[facilities[u]['type']]*cost_orig[i][j]], 
                                lb=[0], 
                                ub=[cplex.infinity], 
                                types=model.variables.type.binary,
                                names=["w_"+str(s1)+"_"+str(s2)+"_"+str(u)+"_"+str(i)+"_"+str(j)])        

    for e in edges:
        for i in G.nodes():
                for j in G.nodes():
                    for u in facilities:
                        (s1,s2)=e                        
                        ind=[]
                        val=[]
                        rhs=0
                        ind.append("w_"+str(s1)+"_"+str(s2)+"_"+str(u)+"_"+str(i)+"_"+str(j))
                        val.append(1)
                        ind.append("Z_out_"+str(s2)+"_"+str(u)+"_"+str(j))
                        val.append(-1)
                        ind.append("Z_out_"+str(s1)+"_"+str(u)+"_"+str(i))
                        val.append(1)
                        model.linear_constraints.add(lin_expr=[cplex.SparsePair(ind, val)], senses='G',rhs=[rhs])
    
    
    
    print G.nodes()
    raw_input("Enter")
    model.objective.set_sense( model.objective.sense.minimize )
    model.set_problem_type( model.problem_type.MILP )
    model.solve()
    model.write("Facility_location",'lp')
    print 'Solved!!!'
    
    for s in scenarios:
        for u in facilities:
            for i in G.nodes():
                if model.solution.get_values("Y_"+str(s)+"_"+str(u)+"_"+str(i))==1:
                    print "Y_"+str(s)+"_"+str(u)+"_"+str(i)
    
    for s in scenarios:
        for u in facilities:
            for i in G.nodes():
                for j in G.nodes():
                    if model.solution.get_values("w_"+str(s)+"_"+str(u)+"_"+str(i)+"_"+str(j))==1:
                        print "w_"+str(s)+"_"+str(u)+"_"+str(i)+"_"+str(j),  model.solution.get_values("w_"+str(s)+"_"+str(u)+"_"+str(i)+"_"+str(j)),"Z_out_"+str(s)+"_"+str(u)+"_"+str(j),int(model.solution.get_values("Z_out_"+str(s)+"_"+str(u)+"_"+str(j))), "Z_in_"+str(s)+"_"+str(u)+"_"+str(i),int(model.solution.get_values("Z_in_"+str(s)+"_"+str(u)+"_"+str(i)))  

    
    print 'Solution: ', model.solution.get_status_string()
    print model.solution.get_objective_value()   

   
facility_location()
