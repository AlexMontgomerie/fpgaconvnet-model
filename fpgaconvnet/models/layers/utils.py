from functools import reduce

def balance_module_rates(rate_graph):

    rate_ratio = [ abs(rate_graph[i,i+1]/rate_graph[i,i]) for i in range(rate_graph.shape[0]) ]

    for i in range(1,rate_graph.shape[0]):
        # start from end
        layer = rate_graph.shape[0]-i

        if abs(rate_graph[layer,layer]) > abs(rate_graph[layer-1,layer]):
            # propogate forward
            for j in range(layer,rate_graph.shape[0]):
                    if(abs(rate_graph[j,j]) <= abs(rate_graph[j-1,j])):
                        break
                    rate_graph[j,j]   = abs(rate_graph[j-1,j])
                    rate_graph[j,j+1] = -rate_graph[j,j]*rate_ratio[j]

        elif abs(rate_graph[layer,layer]) < abs(rate_graph[layer-1,layer]):
            # propogate backward
            for j in range(0,layer):
                    if(abs(rate_graph[layer-j,layer-j]) >= abs(rate_graph[layer-j-1,layer-j])):
                        break
                    rate_graph[layer-j-1,layer-j]   = -abs(rate_graph[layer-j,layer-j])
                    rate_graph[layer-j-1,layer-j-1] = -rate_graph[layer-j-1,layer-j]/rate_ratio[layer-j-1]
    return rate_graph

def get_factors(n):
    return list(set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

