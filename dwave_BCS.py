import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


t = 1.0
tprime = -0.3 * t

Tc = 0.025 * t

def epsilon_k(kx,ky):
    #e_k = 4. * ( t + tprime ) - 2. * t * ( np.cos(kx) + np.cos(ky) ) - 2. * tprime * ( np.cos(kx + ky) + np.cos(kx - ky) )
    e_k = ( 4. * ( t + tprime ) ) + ( -2. * t * ( np.cos(kx) + np.cos(ky) ) ) + ( - 4. * tprime * np.cos(kx) * np.cos(ky) )
    return e_k

def test_epsilon_k():
    L_k = 100
    kx = np.linspace(-np.pi,np.pi,L_k,endpoint=False)
    ky = np.linspace(-np.pi,np.pi,L_k,endpoint=False)
    X, Y = np.meshgrid(kx, ky)
    Z = epsilon_k(X,Y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel(r'$k_x/a$')
    ax.set_ylabel(r'$k_y/a$')
    ax.set_zlabel(r'$\epsilon_{\vec{k}}$')

    plt.show()
            
def Fermi_surface():
    L_k = 100
    kx = np.linspace(-np.pi,np.pi,L_k,endpoint=False)
    ky = np.linspace(-np.pi,np.pi,L_k,endpoint=False)
    X, Y = np.meshgrid(kx, ky)
    Z = epsilon_k(X,Y)

    epsilon_F = 1.7
    cs = plt.contour(X, Y, Z, levels=[epsilon_F], linestyles = '--')
    cs.cmap.set_over('white')
    cs.cmap.set_under('white')
    cs.changed()

    plt.xlabel(r'$k_x/a$')
    plt.ylabel(r'$k_y/a$')
  
    plt.show()

def Delta_k(kx,ky,Delta):
    D_k = Delta * ( np.cos(kx) - np.cos(ky) )
    return D_k

def Delta_k_T(kx,ky,Delta,T):
    ##D_k_T = Delta * ( np.cos(kx) - np.cos(ky) ) * np.sqrt(1-(T/Tc)**2) * np.heaviside (Tc-T, 1) 
    D_k_T = np.where(Tc < T, 0, Delta * ( np.cos(kx) - np.cos(ky) ) * np.sqrt(np.maximum((1-((T/Tc)**2)),0.)))
    return D_k_T

def test_Delta_k():
    Delta = 10.0
    L_k = 100
    kx = np.linspace(-np.pi,np.pi,L_k,endpoint=False)
    ky = np.linspace(-np.pi,np.pi,L_k,endpoint=False)
    X, Y = np.meshgrid(kx, ky)
    Z = Delta_k(X,Y,Delta)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel(r'$k_x/a$')
    ax.set_ylabel(r'$k_y/a$')
    ax.set_zlabel(r'$\Delta_{\vec{k}}$')

    plt.show()
            
def density_no_BZ_symmetry(mu, beta, Delta, L_k):
    integral = 0.0

    for kx_i in np.linspace(-np.pi,np.pi,L_k,endpoint=False):
        for ky_i in np.linspace(-np.pi,np.pi,L_k,endpoint=False):
            eps_k = epsilon_k(kx_i,ky_i)
            Del_k = Delta_k(kx_i,ky_i,Delta)
            xchsi_k = eps_k - mu
            E_k = np.sqrt(xchsi_k**2 + Del_k**2)
            integrand = ( xchsi_k / E_k ) * np.tanh( beta * E_k / 2.)
            integral = integral + integrand
    
    density = 1.0 - ( (1. / L_k**2) * integral )

    return density

def compute_density_as_func_of_T_no_symmetry(mu, Delta, L_k):
    T_vals = []
    density_vals = []
    for T_i in np.arange(0.001, 2/10, 0.001):
        T_vals.append(T_i)
        beta_i = 1/T_i
        density_vals.append(density_no_BZ_symmetry(mu,beta_i,Delta, L_k))

    return T_vals, density_vals

def compute_density_as_func_of_Delta_no_symmetry(mu, beta):
    Delta_vals = []
    density_vals = []
    for Delta_i in np.arange(0, 100, 1):
        Delta_vals.append(Delta_i)
        density_vals.append(density_no_BZ_symmetry(mu,beta,Delta_i))
     
    plt.plot(Delta_vals, density_vals)

    plt.xlabel(r'$\Delta/t$')
    plt.ylabel(r'$\langle n \rangle (\Delta)$')

    plt.show()

def plt_reduced_BZ():
    a1 = np.array([np.pi,0]) 
    a2 = np.array([0,np.pi]) 

    for y in range(1,L_k+1):
        k_y = (2*y-1) / (2.*L_k)
        for x in range(y,L_k+1):
            k_x = (2*x-1) / (2.*L_k)
            k_xy = k_x * a1 + k_y * a2
            plt.scatter(k_xy[0], k_xy[1])
    
    plt.show()

def density_using_BZ_symmetry(mu, beta, Delta, L_k):
    T = 1. / beta

    a1 = np.array([np.pi,0]) 
    a2 = np.array([0,np.pi]) 

    integral = 0.0
    for y in range(1,L_k+1):
        k_y = (2*y-1) / (2.*L_k)
        for x in range(y,L_k+1):
            k_x = (2*x-1) / (2.*L_k)
            k_xy = k_x * a1 + k_y * a2

            eps_k = epsilon_k(k_xy[0],k_xy[1])
            #Del_k = Delta_k(k_xy[0],k_xy[1],Delta)
            Del_k = Delta_k_T(k_xy[0],k_xy[1],Delta, T)
            xchsi_k = eps_k - mu
            E_k = np.sqrt(xchsi_k**2 + Del_k**2)

            if x == y:
                integrand = 0.5 * ( xchsi_k / E_k ) * np.tanh( beta * E_k / 2.)
            else:
                integrand = ( xchsi_k / E_k ) * np.tanh( beta * E_k / 2.)
            
            integral = integral + integrand
             
    total_integral = 8. * ( 1.0 / ( 4.0 * L_k**2 ) ) * integral
    
    density = 1.0 - total_integral 

    return density

def compute_density_as_func_of_T_using_BZ_symmetry(mu, Delta, L_k):
    T_vals = []
    density_vals = []
    for T_i in np.arange(0.00001, 1/10, 0.001):
        T_vals.append(T_i)
        beta_i = 1/T_i
        density_vals.append(density_using_BZ_symmetry(mu, beta_i, Delta, L_k))

    return T_vals, density_vals




def main():
    #test_epsilon_k()
    #Fermi_surface()
    #sys.exit()
    #test_Delta_k()

    #Plot n(T) without using BZ symmetry
    #Delta = 0.0
    #mu = -0.5
    #T_vals, density_vals = compute_density_as_func_of_T_no_symmetry(mu, Delta, 400)
    #plt.plot(T_vals, density_vals, ls = '-', label = r'$\mu = -0.5, \Delta = 0$')
    #plt.show()


    #plot n(T,Delta) using BZ symmetry

    #n(T)
    ##Examples Steve suggested
    #Example 1
    #Delta = 0.0
    #mu = -0.3
    #T_vals, density_vals = compute_density_as_func_of_T_using_BZ_symmetry(mu, Delta, 300)
    #plt.plot(T_vals, density_vals, ls = '--', label = r'$\mu = -0.3, \Delta = 0$')
    #Example 2
    #Delta = 0.0
    #mu = -0.5
    #T_vals, density_vals = compute_density_as_func_of_T_using_BZ_symmetry(mu, Delta, 150)
    #plt.plot(T_vals, density_vals, ls = '--', label = r'$\mu = -0.5, \Delta = 0$')
    #Example 3
    #mu = 0.0
    #Delta = 0.5
    #T_vals, density_vals = compute_density_as_func_of_T_using_BZ_symmetry(mu, Delta, 300)
    #plt.plot(T_vals, density_vals, ls = '--', label = r'$\mu = 0, \Delta = 0.5$')




    #mu(T)
    ##checks
    #T_list = np.arange(0.00001, 1/10, 0.001)

    ##Input parameters for calculation of mu_n(T)
    #Delta = 0.05

    #p = 1. / 8.
    ##n_input = 1. - p
    ##n_input = 0.849
    ##n_input = 0.7740855853966081
    #n_input = 0.7365773889274212


    ##Invert function to get mu_n(T):
    #mu_list = np.linspace(-0.7,0.3,20)
    #n_list = np.array([n_input])
    #mu_target = np.zeros((len(n_list),len(T_list)))

    #for idx_T, T in enumerate(T_list):
    #    beta = 1. / T
    #    n_list_aux = []
    #    for mu in mu_list:
    #        density = density_using_BZ_symmetry(mu,beta,Delta,50)
    #        n_list_aux.append(density)
    #    mu_of_n = interpolate.interp1d(n_list_aux, mu_list, kind='linear')
    #    for idx_n, n in enumerate(n_list):
    #        mu_target[idx_n, idx_T] = mu_of_n(n) 


    ##Input parameters for calculation of n_mu(T)
    ##mu = -0.3
    #mu = -0.5


    #Check:
    #T_input = 0.04
    #n_input = density_using_BZ_symmetry(mu,1./T_input,Delta,50)

    #print('Check:')
    #print(r'n = ')
    #print(n_input) 
    ##print(r'at mu=-0.3 and T/t = 0.04')
    #print(r'at mu=-0.5 and T/t = 0.04')
    
    #print('against')

    #print(r'mu = ')
    #print(mu_target[0,40])
    #print(r'at n = ')
    #print(n_input)
    #print(r'and T/t = 0.04')












    #main1 - mu(T) for paper

    #T_list = np.arange(0.00001, 3.*Tc, 0.001)

    ##Input parameters for calculation of mu_n(T)
    #Delta = 0.1

    #p = 0.2
    #n_input = 1. - p




    #Get an idea of the range of mu to interpolate over
    ##density_vals = density_using_BZ_symmetry(-1.05,1/T_list,Delta,400)
    ##plt.plot(T_list, density_vals, ls = '-', label = r'$n(T)$ at $\mu = -1.05$')
    ##density_vals = density_using_BZ_symmetry(-1.1,1/T_list,Delta,400)
    ##plt.plot(T_list, density_vals, ls = '-', label = r'$n(T)$ at $\mu = -1.1$')

    ##plt.xlabel(r'$T/t$')
    ##plt.ylabel(r'$n(T)$')

    ##plt.legend()

    ##plt.savefig('mu_range_for_desired_density.pdf')
    ##plt.show()

    ##sys.exit()




    ##Invert function to get mu_n(T):
    #mu_list = np.linspace(-0.7,0.3,20)
    #mu_list = np.linspace(-1.1,-1.05,20)
    #n_list = np.array([n_input])
    #mu_target = np.zeros((len(n_list),len(T_list)))

    #for idx_T, T in enumerate(T_list):
    #    beta = 1. / T
    #    n_list_aux = []
    #    for mu in mu_list:
    #        density = density_using_BZ_symmetry(mu,beta,Delta,50)
    #        n_list_aux.append(density)
    #    mu_of_n = interpolate.interp1d(n_list_aux, mu_list, kind='linear')
    #    for idx_n, n in enumerate(n_list):
    #        mu_target[idx_n, idx_T] = mu_of_n(n) 


    ##check -- works
    ##print(T_list[24])
    ##print(mu_target[0,24])
    ##print(density_using_BZ_symmetry(mu_target[0,24],1./T_list[24],Delta,50))


    #plt.plot(T_list,mu_target[0,:], ls = '-')


    #plt.title(r'$T_c = $' + str(Tc) + r'$t, \Delta = $' + str(Delta) + r'$t, t^\prime = -0.3t$')

    #plt.xlabel(r'$T/t$')
    #plt.ylabel(r'$\mu(T)[t]$')

    #plt.tight_layout()
    #plt.savefig('fig_mu(T)_units_of_t.pdf')
    #plt.show()
    #plt.close()




    #plt.plot(T_list*0.4*1.16*10**4,mu_target[0,:]*0.4, ls = '-')


    #plt.title(r'$T_c = 116 \mathrm{K}, \Delta = 0.04 \mathrm{eV}, t = 0.4 \mathrm{eV}, t^\prime = -0.3t$')

    #plt.xlabel(r'$T [\mathrm{K}]$')
    #plt.ylabel(r'$\mu(T)[\mathrm{eV}]$')

    #plt.tight_layout()
    #plt.savefig('fig_mu(T)_units_of_experiment.pdf')
    #plt.show()
    #plt.close()




    #plt.plot(T_list*0.4*1.16*10**4,(mu_target[0,:]-mu_target[0,0])*0.4/mu_target[0,0], ls = '-')


    #plt.title(r'$T_c = 116 \mathrm{K}, \Delta = 0.04 \mathrm{eV}, t = 0.4 \mathrm{eV}, t^\prime = -0.3t$')

    #plt.xlabel(r'$T [\mathrm{K}]$')
    #plt.ylabel(r'$\delta\{\mu(T)\}/\mu(T\approx 0)[\mathrm{eV}]$')

    #plt.tight_layout()
    #plt.savefig('fig_change_mu(T)_units_of_experiment.pdf')
    #plt.show()
    #plt.close()
































    #main2 - mu(T) for paper

    T_list = np.arange(0.00001, 3.*Tc, 0.001)


    ##Input parameters for calculation of mu_n(T)
    #Delta = 0.0

    #p = 0.2
    #n_input = 1. - p




    ##Invert function to get mu_n(T):
    #mu_list = np.linspace(1.698,1.717,30)
    #n_list = np.array([n_input])
    #mu_target = np.zeros((len(n_list),len(T_list)))

    #for idx_T, T in enumerate(T_list):
    #    beta = 1. / T
    #    n_list_aux = []
    #    for mu in mu_list:
    #        density = density_using_BZ_symmetry(mu,beta,Delta,400)
    #        n_list_aux.append(density)
    #    mu_of_n = interpolate.interp1d(n_list_aux, mu_list, kind='linear')
    #    for idx_n, n in enumerate(n_list):
    #        mu_target[idx_n, idx_T] = mu_of_n(n) 


    ##check -- works
    #print(T_list[14])
    #print(mu_target[0,14])
    #print(density_using_BZ_symmetry(mu_target[0,14],1./T_list[14],Delta,400))
    ##sys.exit()


    #epsilon_F = mu_target[0,0]
    #print(epsilon_F)


    ##plt.plot(T_list/Tc, mu_target[0,:]/epsilon_F, ls = '--', lw = 2.5, color='k', label = r'$\Delta = 0$', zorder = 2)
    #plt.plot(T_list/Tc, mu_target[0,:], ls = '--', lw = 2.5, color='black', label = r'$\Delta = 0$', zorder = 2)
    #plt.show()


    #for idx_T, T in enumerate(T_list):
    #    print(T_list[idx_T], "\t", mu_target[0,idx_T])
    #sys.exit()








    ##Input parameters for calculation of mu_n(T)
    Delta = 0.05

    p = 0.2
    n_input = 1. - p




    ##Invert function to get mu_n(T):
    mu_list = np.linspace(1.700,1.716,30)
    n_list = np.array([n_input])
    mu_target = np.zeros((len(n_list),len(T_list)))

    for idx_T, T in enumerate(T_list):
        beta = 1. / T
        n_list_aux = []
        for mu in mu_list:
            density = density_using_BZ_symmetry(mu,beta,Delta,400)
            n_list_aux.append(density)
        mu_of_n = interpolate.interp1d(n_list_aux, mu_list, kind='linear')
        for idx_n, n in enumerate(n_list):
            mu_target[idx_n, idx_T] = mu_of_n(n) 


    ##check -- works
    print(T_list[13])
    print(mu_target[0,13])
    print(density_using_BZ_symmetry(mu_target[0,13],1./T_list[13],Delta,400))
    ##sys.exit()

   
    ##plt.plot(T_list/Tc, mu_target[0,:]/epsilon_F, ls = '-', lw = 2.5, color='k', label = r'$\Delta = 0.1t$', zorder = 1)
    plt.plot(T_list/Tc, mu_target[0,:], ls = '-', lw = 2.5, color='black', label = r'$\Delta = 0.5t$', zorder = 1)
    plt.show()


    for idx_T, T in enumerate(T_list):
        print(T_list[idx_T], "\t", mu_target[0,idx_T])
    sys.exit()








    ##Input parameters for calculation of mu_n(T)
    Delta = 0.5

    p = 0.2
    n_input = 1. - p




    ##Invert function to get mu_n(T):
    mu_list = np.linspace(1.701,1.750,30)
    n_list = np.array([n_input])
    mu_target = np.zeros((len(n_list),len(T_list)))

    for idx_T, T in enumerate(T_list):
        beta = 1. / T
        n_list_aux = []
        for mu in mu_list:
            density = density_using_BZ_symmetry(mu,beta,Delta,400)
            n_list_aux.append(density)
        mu_of_n = interpolate.interp1d(n_list_aux, mu_list, kind='linear')
        for idx_n, n in enumerate(n_list):
            mu_target[idx_n, idx_T] = mu_of_n(n) 


    ##check -- works
    print(T_list[29])
    print(mu_target[0,29])
    print(density_using_BZ_symmetry(mu_target[0,29],1./T_list[29],Delta,150))
    #sys.exit()

   
    #plt.plot(T_list/Tc, mu_target[0,:]/epsilon_F, ls = '-', lw = 2.5, color='k', label = r'$\Delta = 0.1t$', zorder = 1)
    plt.plot(T_list/Tc, mu_target[0,:], ls = '-', lw = 2.5, color='black', label = r'$\Delta = 0.5t$', zorder = 1)
    plt.show()


    for idx_T, T in enumerate(T_list):
        print(T_list[idx_T], "\t", mu_target[0,idx_T])
    sys.exit()
















    plt.xlabel(r'$T/T_c$')
    #plt.ylabel(r'$\mu(T)/E_F$')
    plt.ylabel(r'$\mu(T)[t]$')


    plt.legend(frameon=False,handlelength=2)


    plt.tight_layout()
    plt.savefig('fig_mu(T)_over_epsilonF.pdf')
    plt.show()


main()


