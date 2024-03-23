# discretization method with gaussian elimination

from scipy.linalg import lu, solve
from scipy.sparse import spdiags
from numpy.linalg import cond
from math import sqrt, e, ceil, log

import numpy as np
import matplotlib.pyplot as plt

# constants we need
L = 120 # (in) length
q = 100/12 # (lb/in) intensity of uniform load
E = 3.0e7 # (lb/in^2) modulus of elasticity
S = 1000 # (lb) stress at ends
I = 625 # (in^4) central moment of inertia

# analytical solution
def analy_sol(x):
    global L, q, E, S, I

    c = -(q*E*I)/(S**2)
    a = sqrt(S/(E*I))
    b = -q/(2*S)
    c1 = c*((1-e**(-a*L))/(e**(-a*L)-e**(a*L)))
    c2 = c*((e**(a*L)-1)/(e**(-a*L)-e**(a*L)))

    w = c1*e**(a*x) + c2*e**(-a*x) + b*x*(x-L) + c
    return w



def set_matrix(n, h, x_list):
    global L, q, E, S, I

    A = [[0 for i in range(n-1)] for j in range(n-1)]
    b = [0 for i in range(n-1)]

    # put values in A
    cons_A = 2 + (S/(E*I))*h**2
    main_diagonal = [cons_A for i in range(n-1)]
    upper_diagonal = [-1 for i in range(n-2)]
    lower_diagonal = [-1 for i in range(n-2)]

    # create a tridiagonal matrix
    A = np.diag(main_diagonal) + np.diag(upper_diagonal, k=1) + np.diag(lower_diagonal, k=-1)

    # put values in b
    cons_b = (q/(2*E*I))*h**2
    for i in range(len(b)):
        b[i] = cons_b*x_list[i]*(L-x_list[i])

    A = np.array(A)
    b = np.array(b)

    return A, b

def gaussian_elimination(matrix, vector):
    # LU factorization
    [P, L, U] = lu(matrix)
    # solve for Lc=Pb
    Pb = np.dot(P, vector)
    c = solve(L, Pb)
    # solve for Ux=c
    x = solve(U, c)
    return x



def plot_figure(x_axis, title, xlab, ylab, *args):
    colors = ['steelblue', 'lightcoral', 'navajowhite']
    i=0
    for (num, lab) in args:
        plt.scatter(x_axis, num, label=lab, color=colors[i])
        i += 1

    # Add labels and legend
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()

    # Show plot
    plt.grid(True)
    #plt.show()
    plt.savefig(f"{title}.png")
    print(title, 'done')
    plt.clf()

def plot_error_plots(all_errors, all_x_lists, K):
    plt.figure(figsize=(10, 6))
    colors = ['indianred', 'lightcoral', 'lightsalmon', 'navajowhite', \
              'paleturquoise', 'lightskyblue', 'cornflowerblue', 'royalblue', \
              'steelblue', 'lightseagreen', 'mediumaquamarine', 'lightgreen', 'aquamarine', \
              'thistle']
              
    for k in range(1, K):
        plt.scatter(all_x_lists[k-1], all_errors[k-1], label=f'k={k}', color=colors[k-1])
    plt.yscale('log')
    plt.xlabel('Beam Length (in)')
    plt.ylabel('Error (in)')
    plt.title(f'Error Plots for k=1 to {k}')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.savefig(f"All Errors Log Scale.png")
    print('All Errors Log Scale', 'done')
    plt.clf()



def iteration(k):
    n = int(2**(k+1))
    h = L/n
    # this list contains partitions: x_1 to x_{n-1}
    # notice that list indexing will not correspond to the actual x indexing
    x_list = [h+i*h for i in range(n-1)]

    A, b = set_matrix(n, h, x_list)

    # solution, true value, error
    approx_w = list(gaussian_elimination(A, b))
    true_w = [analy_sol(x_list[i]) for i in range(n-1)]
    error_list = [abs(approx_w[i]-true_w[i]) for i in range(n-1)]
    
    # add the first and last entry for each list
    x_list = [0] + x_list + [L]
    approx_w = [0] + approx_w + [0]
    true_w = [0] + true_w + [0]
    error_list = [0] + error_list + [0]

    all_errors.append(error_list)
    all_x_lists.append(x_list)

    # for debug
    '''print("gaussian elimilation \t analytical solution")
    for i in range(n-1):
        print(approx_w[i], "\t", true_w[i])
    print()'''

    # plot approx vs true solution
    if k==1 or k==2:
        title = f'True and Approximate Solution with k={k}'
        xlabel = 'Beam Length (in)'
        ylabel = 'Deflection (in)'
        plot_figure(x_list, title, xlabel, ylabel, \
                    (approx_w,'approximate'), (true_w,'true'))

    # plot error
    title = f'Errors with k={k}'
    xlabel = 'Beam Length (in)'
    ylabel = 'Error (in)'
    plot_figure(x_list, title, xlabel, ylabel, (error_list,'error'))

    # middle error
    mid_point = ceil(len(error_list)/2)
    mid_error_list.append(error_list[mid_point])

    # condition number
    cond_num = cond(A)
    cond_list.append(cond_num)

    h_list.append(h)


h_list = []
mid_error_list = []
cond_list = []

all_errors = []
all_x_lists = []



def main():
    K = 13
    for k in range(1,K): 
        iteration(k)

    # for debug
    '''print("middle error \t\t condition number")
    for i in range(len(mid_error_list)):
        print(mid_error_list[i], '\t', cond_list[i])
    print()'''

    # plot E
    title = "E = O(h^2)"
    xlabel = 'log(h)'
    xaxis = [log(i) for i in h_list]
    ylabel = 'log(E)'
    yaxis = [log(i) for i in mid_error_list]

    slope_E = (yaxis[-1]-yaxis[0])/(xaxis[-1]-xaxis[0])
    plot_figure(xaxis, title, xlabel, ylabel, \
                (yaxis, f'slope = {slope_E:.3f}'))

    # plot KN
    title = "KN = O(h^{-2})"
    xlabel = 'log(h)'
    xaxis = [log(i) for i in h_list]
    ylabel = 'log(KN)'
    yaxis = [log(i) for i in cond_list]
    
    slope_KN = (yaxis[-1]-yaxis[0])/(xaxis[-1]-xaxis[0])
    plot_figure(xaxis, title, xlabel, ylabel, \
                (yaxis, f'slope = {slope_KN:.3f}'))
    
    plot_error_plots(all_errors, all_x_lists, K)



if __name__ == "__main__":
    main()
