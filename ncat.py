#!/usr/bin/python
"""
Author: Shi Gu
This is the python edition of the network toolbox
Each author for the matlab version is in the doc string of each function
"""
import numpy as np
from scipy import sparse 
# from scipy.sparse import linalg
from numpy import linalg
from scipy.spatial import ConvexHull
import sys
import bct
def zrand(part1,part2):
    '''
    ZRAND     Calculates the z-Rand score and Variation of Information
    distance between a pair of partitions.

    [zRand,SR,SAR,VI] = ZRAND(part1,part2) calculates the z-score of the
    Rand similarity coefficient between partitions part1 and part2. The
    Rand similarity coefficient is an index of the similarity between the
    partitions, corresponding to the fraction of node pairs identified the
    same way by both partitions (either together in both or separate in
    both)

    NOTE: This code requires genlouvain.m to be on the MATLAB path

    Inputs:     part1,  | Partitions that are being
                part2,  | compared with one another

    Outputs:    zRand,  z-score of the Rand similarity coefficient
                SR,     Rand similarity coefficient
                SAR,    Adjusted Rand similarity coefficient
                VI,     Variation of information

    Amanda L. Traud, Eric D. Kelsic, Peter J. Mucha, and Mason A. Porter,
    "Comparing Community Structure to Characteristics in Online Collegiate
    Social Networks," SIAM Review 53, 526-543 (2011).
    '''
    if part1.shape[0] == 1:
        part1 = part1.T
    if part2.shape[0] == 1:
        part2 = part2.T
    if part1.shape != part2.shape:
        print('ERROR: partitions not of equal length')
        return
    '''
    Generate contingency table and calculate row/column marginals
    '''
    part1 = np.asarray(part1).flatten()
    part2 = np.asarray(part2).flatten()
    partSize1 = len(set(part1)) + 1
    partSize2 = len(set(part2)) + 1
    nij=sparse.csr_matrix((np.ones(part1.shape, dtype=int),(part1,part2)), shape=(partSize1, partSize2))
    ni=nij.sum(axis=1)
    nj=nij.sum(axis=0)
    nj=nj.T

    # Identify total number of elements, n, numbers of pairs, M, and numbers of
    # classified-same pairs in each partition, M1 and M2.

    n = part1.shape[0]
    M = np.double(n*(n-1)/2)
    M1 = np.double(((np.multiply(ni,ni)-ni)/2).sum())
    M2 = np.double(((np.multiply(nj,nj)-nj)/2).sum())

    # Pair counting types:

    # same in both
    a = ((nij.multiply(nij)-nij)/2).sum()
    # same in 1, diff in 2'
    b = M1-a
    # same in 2, diff in 1'
    c = M2-a
    # diff in both'
    d = M-(a+b+c)


    # Rand and Adjusted Rand indices:

    SR=(a+d)/(a+b+c+d)
    meana=M1*M2/M
    SAR=(a-meana)/((M1+M2)/2-meana)

    # PS: The adjusted coefficient is calculated by subtracting the expected
    # value and rescale the result by the difference between the maximum
    # allowed value and the mean value

    # CALCULATE VARIANCE OF AND Z-SCORE OF Rand
    # C2=sum(nj.^3);
    # C2=sum(nj.^3);
    # vara = (C1*C2*(n+1) - C1*(4*M2^2+(6*n+2)*M2+n^2+n) - C2*(4*M1^2+(6*n+2)*M1+n^2+n))...
    #     /(n*(n-1)*(n-2)*(n-3)) + M/16 - (4*M1-2*M)^2*(4*M2-2*M)^2/(256*M^2) +...
    #     (8*(n+1)*M1-n*(n^2-3*n-2))*(8*(n+1)*M2-n*(n^2-3*n-2))/(16*n*(n-1)*(n-2)) +...
    #    (16*M1^2-(8*n^2-40*n-32)*M1+n*(n^3-6*n^2+11*n+10))*...
    #     (16*M2^2-(8*n^2-40*n-32)*M2+n*(n^3-6*n^2+11*n+10))/(64*n*(n-1)*(n-2)*(n-3));

    C1=4*((np.power(ni,3)).sum())-8*(n+1)*M1+n*(n*n-3*n-2)
    C2=4*((np.power(nj,3)).sum())-8*(n+1)*M2+n*(n*n-3*n-2)

    # Calculate the variance of the Rand coefficient (a)

    vara = M/16 - np.power((4*M1-2*M),2)*np.power((4*M2-2*M),2)/(256*M*M) + C1*C2/(16*n*(n-1)*(n-2)) + \
    (np.power((4*M1-2*M),2)-4*C1-4*M)*(np.power((4*M2-2*M),2)-4*C2-4*M)/(64*n*(n-1)*(n-2)*(n-3))

    # Calculate the z-score of the Rand coefficient (a)

    zRand=(a-meana)/np.sqrt(vara)


    # CALCULATE THE VARIATION OF INFORMATION

    c1=set(part1);
    c2=set(part2);
    H1=0; H2=0; I=0;
    for i in c1:
        pi=np.double(ni[i])/n
        H1=H1-pi*np.log(pi)
        for j in c2:
            if nij[i,j]:
                pj=np.double(nj[j])/n;
                pij=np.double(nij[i,j])/n;
                I=I+pij*np.log(pij/pi/pj)
    for j in c2:
        pj=np.double(nj[j])/n
        H2=H2-pj*np.log(pj)
    VI = (H1+H2-2*I)
    return (zRand,SR,SAR,VI)

def recruitment(MA,systemByNode):
    '''
    RECRUITMENT      Recruitment coefficient
    R = RECRUITMENT(MA,systemByNode) calculates the recruitment coefficient
    for each node of the network. The recruitment coefficient of a region
    corresponds to the average probability that this region is in the same
    network community as other regions from its own system.
    Inputs:     MA,     Module Allegiance matrix, where element (i,j)
                       represents the probability that nodes i and j
                       belong to the same community
               systemByNode,	vector or cell array containing the system
                       assignment for each node
    Outputs:    R,              recruitment coefficient for each node
    _______________________________________________
    Marcelo G Mattar (08/21/2014)
    '''

    # Initialize output

    R = np.zeros(shape=(systemByNode.size,1), dtype = np.double);


    # Make sure the diagonal of the module allegiance is all nan

    MA = np.double(MA)
    np.fill_diagonal(MA, np.nan)

    # Calculate the recruitment for each node

    for i in range(systemByNode.size):
        thisSystem = systemByNode[i]
        R[i] = np.nanmean(MA[i,systemByNode==thisSystem])
    return R

def promiscuity(S):
    '''
    PROMISCUITY      Promiscuity coefficient
    P = PROMISCUITY(S) calculates the promiscuity coefficient. The
    promiscuity of a temporal or multislice network corresponds to the
    fraction of all the communities in the network in which a node
    participates at least once.
    Inputs:     S,      pxn matrix of community assignments where p is the
                       number of slices/layers and n the number of nodes
    Outputs:    P,      Promiscuity coefficient
    Other m-files required: none
    Subfunctions: none
    MAT-files required: none
    _______________________________________________
    Marcelo G Mattar (08/21/2014)
    '''
    S = np.asarray(S)
    numNodes = np.shape(S)[1]
    numCommunities = len(np.unique(S))
    P = np.zeros((numNodes,1),dtype = np.double)

    for n in range(numNodes):

        # Notice that P=0 if it only participates in one community and P=1 if
        # it participates in every community of the network

        P[n,0] = np.double((len(np.unique(S[:,n]))-1)) / (numCommunities-1)

    return P

def sig_lmc(C, A):
    '''
    This a function that using Lumped Markov chain to calculate
    the significance of clusters in a give communinity structure.
    refer to "Piccardi 2011 in PloS one".
    Here we normalize the original definition of persistence by
    the size of the corresponding cluster to get a better
    INPUT:
        "A" is a N-by-N weighted adjacency matrix
        "C" is a N-by-1 partition(cluster) vector
    OUTPUT:
        normalized persistence probability of all clusters
    '''
    '''
    Transition Matrix
    '''
    C = np.asarray(C)
    A = np.double(A)
    P = np.linalg.solve(np.diag(np.sum(A,axis = 1)),A)
    [eval, evec] = linalg.eigs(P.T, 1)
    if min(evec)<0:
        evec = -evec
    pi = np.double(evec.T)
    num_node = np.double(np.shape(A)[0])
    cl_label = np.double(np.unique(C))
    num_cl = len(cl_label)
    H = np.zeros((num_node, num_cl),dtype = np.double)
    for i in range(num_cl):
        H[:, i] = np.double((C==cl_label[i]))

    # Transition matrix of the lumped Markov chain

    Q = np.dot(np.dot(np.dot(np.linalg.solve(np.diag(np.dot(pi,H).flatten()),H.T),np.diag(pi.flatten())),P),H)
    persistence = np.multiply(np.divide(np.diag(Q), np.sum(H,axis = 0)),np.sum(H))
    return persistence

def q_value(C, A):
    '''
    This is a function that calculates modularity
    '''
    if np.shape(C)[0] == 1:
        C = C.T
    num_node = np.shape(A)[0]
    cl_label = np.unique(C)
    num_cl = len(cl_label)
    cl = np.zeros((num_node, num_cl))
    for i in range(num_cl):
        cl[:, i] = (C==cl_label[i]);
    cl = np.double(cl)
    q_matrix = ((cl.T).dot(A)).dot(cl)

    return q_matrix

def sig_perm_test(C, A, T):
    '''
    This a function that using permutation test to calculate
    the significance of clusters in a give community structure
    INPUT:
    "A" is a N-by-N weighted adjacency matrix
    "C" is a N-by-1 partition(cluster) vector
    "T" is # of random permutations
    OUTPUT:
    "sig" is the significance of all clusters
    "Q" is the modularity of the give partition(cluster)
    "Q_r" are the modularities of all random partitions
    '''
    num_node = np.shape(A)[0]
    num_cl = len(np.unique(C))
    q_matrix = q_value(C,A)
    q_matrix_r = np.zeros(shape = (T, num_cl, num_cl), dtype = np.double)
    for i in range(T):
        c_r = C[np.random.permutation(range(num_node))]
        q_matrix_r[i,:,:] = q_value(c_r, A)
    aver_q = np.zeros(shape = (num_cl,num_cl), dtype = np.double)
    std_q = np.zeros(shape = (num_cl,num_cl), dtype = np.double)
    for i in range(num_cl):
        for j in range(num_cl):
            temp = q_matrix_r[:, i, j].flatten()
            aver_q[i,j] = np.mean(temp)
            std_q[i,j] = np.std(temp)
    sig_matrix = np.divide((q_matrix-aver_q),std_q)
    return sig_matrix

def integration(MA,systemByNode):
    '''
    INTEGRATION      Integration coefficient
    I = INTEGRATION(MA,systemByNode) calculates the integration coefficient
    for each node of the network. The integration coefficient of a region
    corresponds to the average probability that this region is in the same
    network community as regions from other systems.
    Inputs:     MA,     Module Allegiance matrix, where element (i,j)
                        represents the probability that nodes i and j
                        belong to the same community
                        systemByNode,	vector or cell array containing the system
                         assignment for each node
    Outputs:    I,              integration coefficient for each node
    _______________________________________________
    Marcelo G Mattar (08/21/2014)
    '''

    # Initialize output

    I = np.zeros(shape=(len(systemByNode),1), dtype = np.double)

    # Make sure the diagonal of the module allegiance is all nan

    MA = np.double(MA)
    np.fill_diagonal(MA, np.nan)


    # Calculate the integration for each node

    for i in range(systemByNode.size):
        thisSystem = systemByNode[i]
        I[i] = np.nanmean(MA[i,systemByNode!=thisSystem])
    return I

def integration_recruitment(MA, S):
    '''
    Input Module-Allegiance "MA" and community strucutre "S"
    Output Integration and Recruitment
    '''


    # transform S to a column vector

    if min(S) == 1:
        S = S-1
    if np.shape(S)[0] == 1:
        S = S.T
    MA = np.double(MA)
    num_node = len(S)
    num_cl = max(S)+1
    H = np.zeros(shape=(num_node, num_cl), dtype = np.double)
    for i in range(num_cl):
        H[:,i] = (S==i)
    D_H = (H.T).dot(H)

    recruitment = np.zeros(shape = (num_cl, num_cl))
    integration = np.zeros(shape = (num_cl, num_cl))

    D_H_Inv = linalg.inv(D_H)
    recruitment = D_H_Inv.dot(H.T).dot(MA).dot(H).dot(D_H_Inv)
    D = np.diag(np.diag(recruitment))
    D_Inv_Sqr = np.power(D, -0.5)
    integration = D_Inv_Sqr.dot(recruitment).dot(D_Inv_Sqr)
    return (integration,recruitment)

def flexibility(*arg):
    '''
    FLEXIBILITY    Flexibility coefficient
    F = FLEXIBILITY(S, NETTYPE) calculates the flexibility coefficient of
    S. The flexibility of each node corresponds to the number of times that
    it changes module allegiance, normalized by the total possible number
    of changes. In temporal networks, we consider changes possible only
    between adjacent time points. In multislice/categorical networks,
    module allegiance changes are possible between any pairs of slices.
    Inputs:     S,      pxn matrix of community assignments where p is the
                       number of slices/layers and n the number of nodes
            nettype,   string specifying the type of the network:
                       'temp'  temporal network (default)
                       'cat'   categorical network
    Outputs:    F,      Flexibility coefficient
    Other m-files required: none
    Subfunctions: none
    MAT-files required: none
    See also: PROMISCUITY
    _______________________________________________
    Marcelo G Mattar (08/21/2014)
    '''
    # CHECK INPUTS

    if len(arg) < 2:
        S = arg[0]
        nettype = 'temp'
    elif len(arg) > 3:
        raise sys.error( 'To many inputs')
    else:
        S = arg[0]
        nettype = arg[1]

    if not(nettype=='temp' or nettype =='cat'):
        raise sys.error('Expected input ''nettype'' to match ''temp'' or ''cat''')

    if len(np.shape(S)) > 2:
        raise sys.error('S must be a pxn matrix')

    (numSlices, numNodes) = np.shape(S)



    # CALCULATE FLEXIBILITY


    # Pre-allocate totalChanges

    totalChanges = np.zeros(shape=((numNodes,1)), dtype = np.double)
    if nettype == 'temp':

        # only consider adjacent slices

        possibleChanges = numSlices-1
        for t in range(1,numSlices):
            totalChanges = totalChanges + (S[t,:] != S[t-1,:]).T

    elif nettype == 'cat':

        # consider all pairs of slices

        possibleChanges = numSlices*(numSlices-1)
        for s in range(numSlices):
            otherSlices = range(numSlices)

            # all slices but the current one

            otherSlices = otherSlices(otherSlices!=s)
            totalChanges = totalChanges + np.sum(np.tile(S[s,:],(numSlices-1),1) != S[otherSlices,:],axis = 0).T


    # Calculate output

    F = totalChanges/possibleChanges
    return F

def consensus_similarity(C):
    '''
    CONSENSUS_ITERATIVE     Construct a consensus (representative) partition
    using the iterative thresholding procedure
    [consensus, consensus_simm, pairwise_simm] = CONSENSUS_SIMILARITY(C)
    identifies a single representative partition from a set of C partitions
    that is the most similar to the all others. Here, similarity is taken
    to be the z-score of the Rand coefficient (see zrand.m)
    NOTE: This code requires zrand.m to be on the MATLAB path
    Inputs:     C,      pxn matrix of community assignments where p is the
                        number of optimizations and n the number of nodes
    Outputs:    consensus,      consensus partition
                consensus_simm,	average similarity between consensus
                partition and all others
                pairwise_simm,	pairwise similarity matrix
    _______________________________________________
    Marcelo G Mattar (08/21/2014)
    '''

    # number of partitions

    npart = len(C[:,0])


    # Initialize variables

    pairwise_simm = np.zeros(shape=(npart,npart), dtype = np.double)


    # CALCULATE PAIRWISE SIMILARITIES

    for i in range(npart):
        for j in range(i+1,npart):
            pairwise_simm[i,j] = zrand(C[i,:],C[j,:])[0]

    pairwise_simm = pairwise_simm + pairwise_simm.T


    # Average pairwise similarity

    average_pairwise_simm = np.sum(pairwise_simm,axis = 1)/(npart-1)


    # EXTRACT PARTITION MOST SIMILAR TO THE OTHERS

    (X,I) = (np.max(average_pairwise_simm), np.argmax(average_pairwise_simm))
    consensus = C[I,:]
    consensus_simm = X
    return (consensus, consensus_simm, pairwise_simm)

def consensus_iterative(C):
    '''
    CONSENSUS_ITERATIVE     Construct a consensus (representative) partition
    using the iterative thresholding procedure
    [S2 Q2 X_new3 qpc] = CONSENSUS_ITERATIVE(C) identifies a single
    representative partition from a set of C partitions, based on
    statistical testing in comparison to a null model. A thresholded nodal
    association matrix is obtained by subtracting a random nodal
    association matrix (null model) from the original matrix. The
    representative partition is then obtained by using a Generalized
    Louvain algorithm with the thresholded nodal association matrix.
    NOTE: This code requires genlouvain.m to be on the MATLAB path
    Inputs:     C,      pxn matrix of community assignments where p is the
                        number of optimizations and n the number of nodes

    Outputs:    S2,     pxn matrix of new community assignments
                Q2,     associated modularity value
                X_new3, thresholded nodal association matrix
                qpc,    quality of the consensus (lower == better)

    Bassett, D. S., Porter, M. A., Wymbs, N. F., Grafton, S. T., Carlson,
    J. M., & Mucha, P. J. (2013). Robust detection of dynamic community
    structure in networks. Chaos: An Interdisciplinary Journal of Nonlinear
    Science, 23(1), 013142.
    '''


    # Number of partitions

    npart = len(C[:,0])

    # size of the network

    m = len(C[0,:])


    # Initialize


    # Permuted version of C

    C_rand3 = np.zeros(shape = np.shape(C), dtype = np.int)

    # Nodal association matrix for C

    X = np.zeros(shape =(m,m), dtype = np.double)

    # Random nodal association matrix for C_rand3

    X_rand3 = np.zeros(shape =(m,m), dtype = np.double)


    # NODAL ASSOCIATION MATRIX


    # try a random permutation approach

    for i in range(npart):
        pr = np.random.permutation(range(m))

        # C_rand3 is the same as C, but with each row permuted

        C_rand3[i,:] = C[i,pr]


    # Calculate the nodal association matrices X and X_rand3

    for i in range(npart):
        print i
        for k in range(m):
            for p in range(m):

                # element (i,j) indicate the number of times node i and node j
                # have been assigned to the same community

                if C[i,k] == C[i,p]:
                    X[k,p] = X[k,p] + 1
                else:
                    X[k,p] = X[k,p] + 0


                # element (i,j) indicate the number of times node i and node j
                # are expected to be assigned to the same community by chance

                if C_rand3[i,k] == C_rand3[i,p]:
                    X_rand3[k,p] = X_rand3[k,p] + 1
                else:
                    X_rand3[k,p] = X_rand3[k,p]+ 0



    # THRESHOLDING
    # keep only associated assignments that occur more often than expected in
    # the random data

    X_new3 = np.zeros(shape=(m,m), dtype = np.double)
    X_new3[X > np.amax(np.triu(X_rand3,k = 1))] = X[X > np.amax(np.triu(X_rand3, k = 1))]


    # GENERATE THE REPRESENTATIVE PARTITION


    # recompute optimal partition on this new matrix of kept community
    # association assignments

    S2 = np.zeros(shape=(npart, m), dtype = np.int)
    Q2 = np.zeros(shape=(npart,1),  dtype = np.double)
    for i in range(npart):
        print i

        # [S2(i,:) Q2(i)] = multislice_static_unsigned(X_new3,1);

        [S,Q] = bct.community_louvain(W=X_new3,gamma=0)
        S2[i,:] = S
        Q2[i]  = Q

    # define the quality of the consensus

    qpc = np.sum(abs(np.diff(S2, axis=0)))
    return (S2, Q2, X_new3, qpc)

def comm_radius(partitions,locations):
    '''
    function to calcuate the radius
    of detected communities
    Written by Sarah Feldt Muldoon

    Output:
          comm_radius_array   --- an (M+1)x2 array where M is the number
          of communities.  The first column contains the community index and
          the second column contains radius of that community.
          The M+1th entry contains the average community radius of the
          network (denoted as community 0).
    Input:
          'partitions'    --- an Nx1 array where N is the number of nodes
          in the network.  Each entry contains the communitiy index for
          node i (where communities are sequentially indexed as 1:M and M
          is the total number of detected communities).
          'locations'   ---  a Nxdim array where N is the number of nodes and
          dim is the spatial dimensionality of the network (1,2,or 3).  The
          columns contain the (x,y,z) coordinates of the nodes in euclidean
          space
    '''
    number_nodes = len(partitions)
    number_communities = max(partitions)
    comm_radius_array = np.zeros(shape=(number_communities+1,2), dtype = np.double)
    comm_radius_array[:,0] = range(1,number_communities+1)+[0]
    number_nodes_in_communities = np.zeros(shape=(number_communities,1), dtype = np.int)

    # calcuate the radius for each community
    for i in range(1,number_communities+1):
        community_i_nodes = np.where(partitions==i)[0]
        number_nodes_in_i = len(community_i_nodes)
        number_nodes_in_communities[i-1] = number_nodes_in_i
        if number_nodes_in_i >= 2:
            position_vectors_i = locations[community_i_nodes,:]
            std_position_vectors_i = np.std(position_vectors_i, axis = 0)
            radius_i = np.sqrt(np.power(std_position_vectors_i,2).sum())
            comm_radius_array[i-1,1] = radius_i

    # calcuate the averge community radius for the partition
    std_all_position_vectors = np.std(locations,axis = 0)
    radius_all = np.sqrt(np.power(std_all_position_vectors,2).sum(axis=0))
    average_community_radius=1./(number_nodes*radius_all)*\
    np.sum(np.multiply(number_nodes_in_communities.T,comm_radius_array[range(number_communities),1]))
    comm_radius_array[number_communities,1]=average_community_radius

    return comm_radius_array

def tetrahedron_volume(a, b, c, d):
    return np.abs(np.einsum('ij,ij->i', a-d, np.cross(b-d, c-d))) / 6

def triangle_area(a,b,c):
    return np.abs(np.cross(a-b,a-c))/2

def convex_hull_vertices_volume(pts,dim):
    '''
    function return the ConvexHull Index and the Volume
    '''
    ch = ConvexHull(pts)

    simplices = np.column_stack((np.repeat(ch.vertices[0], ch.nsimplex),
                                 ch.simplices))
    tets = ch.points[simplices]
    if dim == 3:
        return (ch.vertices,np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                     tets[:, 2], tets[:, 3])))
    if dim == 2:
        return (ch.vertices,np.sum(triangle_area(tets[:, 0], tets[:, 1],
                                     tets[:, 2])))
    return 0

def comm_spatial_extent(partitions,locations):
    '''
    function to calcuate the spatial extent
    of detected communities
    Written by Sarah Feldt Muldoon
    Output:
          comm_spatial_extent_array   --- an (M+1)x2 array where M is the number
          of communities.  The first column contains the community index and
          the second column contains the spatial extent
          of that community.  The M+1th entry contains the
          spatial extent of the network (denoted as community 0).


    Input:
          'partitions'    --- an Nx1 array where N is the number of nodes
          in the network.  Each entry contains the communitiy index for
          node i (where communities are sequentially indexed as 1:M and M
          is the total number of detected communities).

          'locations'   ---  a Nxdim array where N is the number of nodes and
          dim is the spatial dimensionality of the network (2,or 3).  The
          columns contain the (x,y,z) coordinates of the nodes in euclidean
          space
    '''



    number_nodes=len(partitions)
    number_communities=max(partitions)
    comm_spatial_extent_array=np.zeros(shape=(number_communities+1,2), dtype = np.double)
    comm_spatial_extent_array[:,0] = range(1,number_communities+1)+[0]
    number_nodes_in_communities=np.zeros(shape=(number_communities,1), dtype = np.int)

    dim = np.shape(locations)[1]


    # calcuate the spatial extent for each community
    for i in range(1,number_communities+1):
        community_i_nodes=np.where(partitions==i)[0]
        number_nodes_in_i=len(community_i_nodes)
        number_nodes_in_communities[i-1]=number_nodes_in_i
        if number_nodes_in_i < dim+1:
            volume_i = 0
        else:
            volume_i = convex_hull_vertices_volume(locations[community_i_nodes,:], dim)[1]
        spatial_extent_i = volume_i/number_nodes_in_i
        comm_spatial_extent_array[i-1,1]=spatial_extent_i;


    # calcuate the spatial extent for the entire network
    if number_nodes < dim+1:
        volume_i = 0
    else:
        volume_i = convex_hull_vertices_volume(locations, dim)[1]
    spatial_extent_network=volume_i/number_nodes;
    comm_spatial_extent_array[number_communities,1]=spatial_extent_network;

    return comm_spatial_extent_array

def comm_spatial_diameter(partitions,locations):
    '''
    function to calcuate the spatial diameter
    of detected communities
    Written by Sarah Feldt Muldoon

    Output:
          comm_spatial_diameter_array   --- an (M+1)x2 array where M is the number
          of communities.  The first column contains the community index and
          the second column contains the spatial diameter
          within that community.  The M+1th entry contains the
          spatial diameter of the network (denoted as community 0).
    Input:
          'partitions'    --- an Nx1 array where N is the number of nodes
          in the network.  Each entry contains the communitiy index for
          node i (where communities are sequentially indexed as 1:M and M
          is the total number of detected communities).
          'locations'   ---  a Nxdim array where N is the number of nodes and
          dim is the spatial dimensionality of the network (1,2,or 3).  The
          columns contain the (x,y,z) coordinates of the nodes in euclidean
          space
    '''
    number_nodes=len(partitions)
    number_communities=max(partitions)
    comm_spatial_diameter_array=np.zeros(shape=(number_communities+1,2), dtype = np.double)
    comm_spatial_diameter_array[:,0]=range(1,number_communities+1)+[0]

    # calculate for each community
    for i in range(number_communities):
        community_i_nodes=np.where(partitions==i+1)[0]
        number_nodes_in_i=len(community_i_nodes)
        dist_array=[]
        if number_nodes_in_i >= 2:
            for j in range(number_nodes_in_i):
                node_j=community_i_nodes[j]
                for k in range(j+1,number_nodes_in_i):
                    node_k=community_i_nodes[k]
                    coordinates_j=locations[node_j,:]
                    coordinates_k=locations[node_k,:]
                    print 'Coordinates:'
                    print coordinates_j
                    print coordinates_k
                    dist_jk=linalg.norm(coordinates_j-coordinates_k)
                    dist_array=dist_array+[dist_jk]
            max_dist_i=max(dist_array)
            comm_spatial_diameter_array[i,1]=max_dist_i

        # calculate for the entire network
    dist_array=[]
    for i in range(number_nodes):
        coordinates_i=locations[i,:]
        for j in range(i+1,number_nodes):
            coordinates_j=locations[j,:]
            dist_ij=linalg.norm(coordinates_i-coordinates_j)
            dist_array=dist_array +[dist_ij]
    max_dist_network=max(dist_array)
    comm_spatial_diameter_array[number_communities,1]=max_dist_network
    return comm_spatial_diameter_array

def comm_laterality(partitions,categories):
    '''
    function to calcuate the laterality
    of detected communities
    Written by Sarah Feldt Muldoon
    Output:
          comm_laterality_array   --- an (M+1)x2 array where M is the number
          of communities.  The first column contains the community index and
          the second column contains the laterality of that community.
          The M+1th entry contains the  laterality of the partition of the
          network (denoted as community 0).
    Input:
          'partitions'    --- an Nx1 array where N is the number of nodes
          in the network.  Each entry contains the communitiy index for
          node i (where communities are sequentially indexed as 1:M and M
          is the total number of detected communities).
          'categories'   ---  a Nx1 array where N is the number of nodes and
          each entry is either a '0' or '1' denoting the assignment of each
          node to one of two communities.
    '''
    number_nodes=len(partitions)
    number_communities=max(partitions)
    comm_laterality_array=np.zeros(shape=(number_communities+1,2), dtype=np.double)
    comm_laterality_array[:,0]=range(1,number_communities+1)+[0]
    number_nodes_in_communities=np.zeros(shape=(number_communities,3),dtype=np.int)

    # calcuate the laterality for each community
    for i in range(number_communities):
        community_i_nodes=np.where(partitions==(i+1))[0]
        number_nodes_in_i=len(community_i_nodes)
        number_nodes_in_communities[i,0]=number_nodes_in_i
        categories_i=categories[community_i_nodes]

        nodes_in_comm_0=np.where(categories_i==0)[0]
        number_nodes_in_comm_0=len(nodes_in_comm_0)
        number_nodes_in_communities[i,1]=number_nodes_in_comm_0

        nodes_in_comm_1=np.where(categories_i==1)[0]
        number_nodes_in_comm_1=len(nodes_in_comm_1)
        number_nodes_in_communities[i,2]=number_nodes_in_comm_1

        laterality_i=np.abs(number_nodes_in_comm_0-number_nodes_in_comm_1)/np.double(number_nodes_in_i)
        comm_laterality_array[i,1]=laterality_i

    # calcuate the laterality for the network partition
    # need to calcuated "expected" laterality from each community from surrogate
    # data
    total_nodes_assignments=np.sum(number_nodes_in_communities,axis = 0)
    total_number_in_comm_1=total_nodes_assignments[2]
    number_surrogates=1000
    surrogate_laterality=np.zeros(shape=(number_communities,number_surrogates), dtype = np.double)
    for j in range(number_surrogates):
        randomized_categories=np.zeros(shape=(number_nodes), dtype=np.int)
        rand_perm_assignment=np.random.permutation(range(number_nodes))
        place_in_comm_1=rand_perm_assignment[0:total_number_in_comm_1]
        randomized_categories[place_in_comm_1]=1
        # now calculate the community laterality for the randomized data
        for i in range(number_communities):
            community_i_nodes=np.where(partitions==i+1)[0]
            number_nodes_in_i=len(community_i_nodes)
            randomized_categories_i=randomized_categories[community_i_nodes]

            rand_nodes_in_comm_0=np.where(randomized_categories_i==0)[0]
            rand_number_nodes_in_comm_0=len(rand_nodes_in_comm_0)

            rand_nodes_in_comm_1=np.where(randomized_categories_i==1)[0]
            rand_number_nodes_in_comm_1=len(rand_nodes_in_comm_1)

            rand_laterality_i=np.abs(rand_number_nodes_in_comm_0-rand_number_nodes_in_comm_1)/np.double(number_nodes_in_i)
            surrogate_laterality[i,j]=rand_laterality_i

    expected_comm_laterality=np.sum(surrogate_laterality,axis = 1)/number_surrogates

    network_laterality=1./number_nodes*np.sum(np.multiply(number_nodes_in_communities[:,0]\
    ,comm_laterality_array[0:number_communities,1])-expected_comm_laterality)

    comm_laterality_array[number_communities,1]=network_laterality
    return comm_laterality_array

def comm_ave_pairwise_spatial_dist(partitions,locations):
    '''
    function to calcuate the average pairwise spatial distance between nodes
    within detected communities
    Written by Sarah Feldt Muldoon
    Output:
          comm_ave_pairwise_spatial_dist_array   --- an (M+1)x2 array where M is the number
          of communities.  The first column contains the community index and
          the second column contains the average pairwise spatial distance
          within that community.  The M+1th entry contains the average pairwise
          spatial distance of the network (denoted as community 0).
    Input:
          'partitions'    --- an Nx1 array where N is the number of nodes
          in the network.  Each entry contains the communitiy index for
          node i (where communities are sequentially indexed as 1:M and M
          is the total number of detected communities).
          'locations'   ---  a Nxdim array where N is the number of nodes and
          dim is the spatial dimensionality of the network (1,2,or 3).  The
          columns contain the (x,y,z) coordinates of the nodes in euclidean
          space
    '''


    number_nodes=len(partitions)
    number_communities=max(partitions)

    comm_ave_pairwise_spatial_dist_array=np.zeros(shape=(number_communities+1,2),dtype = np.double)
    comm_ave_pairwise_spatial_dist_array[:,0]=range(1,number_communities+1)+[0]

    # calcuate for each community
    for i in range(number_communities):
        community_i_nodes=np.where(partitions==i+1)[0]
        number_nodes_in_i=len(community_i_nodes)
        dist_array=[]
        if number_nodes_in_i >= 2:
            for j in range(number_nodes_in_i-1):
                node_j=community_i_nodes[j]
                for k in range(j+1,number_nodes_in_i):
                    node_k=community_i_nodes[k]
                    coordinates_j=locations[node_j,:]
                    coordinates_k=locations[node_k,:]
                    dist_jk=linalg.norm(coordinates_j-coordinates_k)
                    dist_array.append(dist_jk)

            ave_dist_i=np.mean(dist_array)
            comm_ave_pairwise_spatial_dist_array[i,1]=ave_dist_i

    # calculate for the network
    dist_array=[]
    for i in range(number_nodes-1):
        coordinates_i=locations[i,:]
        for j in range(i+1,number_nodes):
            coordinates_j=locations[j,:]
            dist_ij=linalg.norm(coordinates_i-coordinates_j)
            dist_array.append(dist_ij)

    ave_dist_network=np.mean(dist_array)
    comm_ave_pairwise_spatial_dist_array[number_communities,1]=ave_dist_network
    return comm_ave_pairwise_spatial_dist_array
def genlouvain(W, ci=None, B='modularity', seed=None):
    '''
    The optimal community structure is a subdivision of the network into
    nonoverlapping groups of nodes which maximizes the number of within-group
    edges and minimizes the number of between-group edges.
    This function is a fast an accurate multi-iterative generalization of the
    louvain community detection algorithm. This function subsumes and improves
    upon modularity_[louvain,finetune]_[und,dir]() and additionally allows to
    optimize other objective functions (includes built-in Potts Model i
    Hamiltonian, allows for custom objective-function matrices).
    Parameters
    ----------
    W : NxN np.array
        directed/undirected weighted/binary adjacency matrix
    gamma : float
        resolution parameter. default value=1. Values 0 <= gamma < 1 detect
        larger modules while gamma > 1 detects smaller modules.
        ignored if an objective function matrix is specified.
    ci : Nx1 np.arraylike
        initial community affiliation vector. default value=None
    B : str | NxN np.arraylike
        string describing objective function type, or provides a custom
        objective-function matrix. builtin values 'modularity' uses Q-metric
        as objective function, or 'potts' uses Potts model Hamiltonian.
        Default value 'modularity'.
    seed : int | None
        random seed. default value=None. if None, seeds from /dev/urandom.
    Returns
    -------
    ci : Nx1 np.array
        final community structure
    q : float
        optimized q-statistic (modularity only)
    '''
    np.random.seed(seed)

    n = len(W)
    s = np.sum(W)


    if ci is None:
        ci = np.arange(n) + 1
    else:
        if len(ci) != n:
            raise ValueError('initial ci vector size must equal N')
        _, ci = np.unique(ci, return_inverse=True)
        ci += 1
    Mb = ci.copy()

    if B == 'modularity':
        B = W
    else:
        try:
            B = np.array(B)
        except:
            raise ValueError('unknown objective function type')

        if B.shape != W.shape:
            raise ValueError('objective function matrix does not match '
                                'size of adjacency matrix')
        if not np.allclose(B, B.T):
            print ('Warning: objective function matrix not symmetric, '
                   'symmetrizing')
            B = (B + B.T) / 2

    Hnm = np.zeros((n, n))
    for m in range(1, n + 1):
        Hnm[:, m - 1] = np.sum(B[:, ci == m], axis=1)  # node to module degree
    H = np.sum(Hnm, axis=1)  # node degree
    Hm = np.sum(Hnm, axis=0)  # module degree

    q0 = -np.inf
    # compute modularity
    q = np.sum(B[np.tile(ci, (n, 1)) == np.tile(ci, (n, 1)).T]) / s

    first_iteration = True

    while q - q0 > 1e-10:
        it = 0
        flag = True
        while flag:
            it += 1
            if it > 1000:
                raise ValueError('Modularity infinite loop style G. '
                                    'Please contact the developer.')
            flag = False
            for u in np.random.permutation(n):
                ma = Mb[u] - 1
                dQ = Hnm[u, :] - Hnm[u, ma] + B[u, u]  # algorithm condition
                dQ[ma] = 0

                max_dq = np.max(dQ)
                if max_dq > 1e-10:
                    flag = True
                    mb = np.argmax(dQ)

                    Hnm[:, mb] += B[:, u]
                    Hnm[:, ma] -= B[:, u]  # change node-to-module strengths

                    Hm[mb] += H[u]
                    Hm[ma] -= H[u]  # change module strengths

                    Mb[u] = mb + 1

        _, Mb = np.unique(Mb, return_inverse=True)
        Mb += 1

        M0 = ci.copy()
        if first_iteration:
            ci = Mb.copy()
            first_iteration = False
        else:
            for u in range(1, n + 1):
                ci[M0 == u] = Mb[u - 1]  # assign new modules

        n = np.max(Mb)
        b1 = np.zeros((n, n))
        for i in range(1, n + 1):
            for j in range(i, n + 1):
                # pool weights of nodes in same module
                bm = np.sum(B[np.ix_(Mb == i, Mb == j)])
                b1[i - 1, j - 1] = bm
                b1[j - 1, i - 1] = bm
        B = b1.copy()

        Mb = np.arange(1, n + 1)
        Hnm = B.copy()
        H = np.sum(B, axis=0)
        Hm = H.copy()

        q0 = q
        q = np.trace(B) / s  # compute modularity

    return ci, q
def multislice_static_unsigned(A, gplus):
   '''
   INPUTS
   A is the (weighted) connectivity matrix
   it is assumsed that all values of the connectivity matrix are positive
   Gplus is the resolution parameter. If unsure, use default value of 1.

   OUTPUTS
   S is the partition (or community assignment of all nodes to communities)
   Q is the modularity of the (optimal) partition
   lAlambda is the effective fraction of antiferromagnetic edges (see Onnela
   et al. 2011 http://arxiv.org/pdf/1006.5731v1.pdf)

   This code uses the Louvain heuristic

   DB 2012
   '''
   Aplus=A
   Aplus[A<0]=0
   twom = np.sum(Aplus)
   P = np.outer(np.sum(Aplus, axis=1), np.sum(Aplus, axis=0)) / twom
   B=A-gplus*P;
   lAlambda = np.sum((np.divide(A,P)<gplus));

   S, Q = genlouvain(B)
   Q=Q/twom
   return S,Q,lAlambda

