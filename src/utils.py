from random import random
import numpy as np
from scipy import integrate
from tqdm import tqdm
from math import floor
import scipy.stats as stat


class binomial_tree:
    """
    Defines a binomial tree
    """
    def __init__(self, r, p, T, Nt, u, d, S0, payoff_fct = None, american = False, simple_interest = True, disable_progress_bar = False):
        self.p = p
        self.r = r
        self.u = u
        self.d = d
        self.T = T
        self.Nt = Nt
        self.S0 = S0
        self.dt = self.T/self.Nt
        self.disable_progress_bar = disable_progress_bar
        if simple_interest:
            self.R = 1. + self.dt * self.r
        else:
            self.R = np.exp(self.dt * self.r)
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))
        assert self.d<1 + self.r * self.dt * self.T, 'there is ABRITRAGE!'
        assert self.u>1 + self.r * self.dt * self.T, 'there is ABRITRAGE!'
        self.q = (self.R - self.d)/(self.u - self.d)
        if payoff_fct is not None:
            self.payoff_fct = payoff_fct
            self.asset_prices = self.compute_asset_prices_upper_triangular()
            if american:
                self.derivative_prices, self.exercise_early = self.compute_american_derivative_prices_upper_triangular()
            else:
                self.derivative_prices = self.compute_european_derivative_prices_upper_triangular()
            self.derivative_price_at_zero = self.derivative_prices[0,0]

    def simulate(self, nsims = 1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = self.S0
        for t in range(self.Nt):
            randomness = np.random.binomial(n=1, p=self.p, size = (nsims,))
            x[t+1,:] = x[t,:] * self.u * randomness + x[t,:] * self.d * (1 - randomness)
        return x
    
    def compute_asset_prices_upper_triangular(self):
        asset_prices = np.zeros((self.Nt+1,self.Nt+1))
        for j in range(self.Nt+1):
            for i in range(j+1):
                asset_prices[i,j] = self.S0*self.u**(j-i)*self.d**(i)
        return asset_prices
    
    def compute_one_step_derivative_price(self, Vu, Vd):
        price = 1./self.R * (self.q * Vu + (1.-self.q)*Vd )
        return price
    
    def compute_european_derivative_prices_upper_triangular(self):
        derivative_prices = np.zeros((self.Nt+1,self.Nt+1))
        derivative_prices[:] = np.nan
        derivative_prices[:,-1] = self.payoff_fct(self.asset_prices[:,-1])
        for j in tqdm(range(self.Nt), disable= self.disable_progress_bar):
            col = self.Nt - j - 1
            for i in range(col+1):
                Vu = derivative_prices[i, col+1]
                Vd = derivative_prices[i+1, col+1]
                derivative_prices[i,col] = self.compute_one_step_derivative_price(Vu,Vd)
        return derivative_prices
    
    def compute_american_derivative_prices_upper_triangular(self):
        derivative_prices = np.zeros((self.Nt+1,self.Nt+1))
        derivative_prices[:] = np.nan
        exercise_early = np.zeros((self.Nt+1,self.Nt+1))
        derivative_prices[:,-1] = self.payoff_fct(self.asset_prices[:,-1])
        for j in tqdm(range(self.Nt), disable= self.disable_progress_bar):
            col = self.Nt - j - 1
            for i in range(col+1):
                Vu = derivative_prices[i, col+1]
                Vd = derivative_prices[i+1, col+1]
                derivative_prices[i,col] = self.compute_one_step_derivative_price(Vu, Vd)
                if derivative_prices[i,col] != derivative_prices[i,col]:
                    print('NaN: Vu=', Vu,' Vd=',Vd,'i=',i,' col =',col)
                if self.asset_prices[i,col] != self.asset_prices[i,col]:
                    print('NaN: Vu=', Vu,' Vd=',Vd,'i=',i,' col =',col)
                exercise_early[i,col] = ( np.round(self.payoff_fct(self.asset_prices[i,col]),15) 
                                          > np.round(derivative_prices[i,col],15) ).astype(bool)
                derivative_prices[i,col] = (derivative_prices[i,col] * (1-exercise_early[i,col]) 
                                            + self.payoff_fct(self.asset_prices[i,col])* exercise_early[i,col])
        return derivative_prices, exercise_early
    
    

class RandomWalk:
    """
    Defines and simulates a Random Walk
    """
    def __init__(self, p, T, Nt):
        self.p = p
        self.T = T
        self.Nt = Nt
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))
        assert self.Nt>self.T, 'The number of steps should be greater than the terminal time'
        
    def simulate(self, nsims = 1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = 0.
        errs = (np.random.rand(self.Nt, nsims) <= self.p) * 2. - 1.
        for t in range(self.Nt):
            if floor(self.timesteps[t+1]) > floor(self.timesteps[t]):
                x[t + 1,:] = x[t,:] + errs[t,:]
            else:
                x[t + 1,:] = x[t,:]
        return x


class BrownianMotion:
    """
    Defines and simulates a standard Brownian motion
    """
    def __init__(self, T, Nt):
        self.T = T
        self.Nt = Nt
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))
        
    def simulate(self, nsims = 1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = 0.
        dt = self.T/(self.Nt)
        errs = np.random.randn(self.Nt, nsims)
        for t in range(self.Nt):
            x[t + 1,:] = x[t,:] + np.sqrt(dt) * errs[t,:]
        return x

    
class ArithmeticBrownianMotion:
    """
    Model parameters for the environment.
    """
    def __init__(self, x0 = 0, mu = 0.5, sigma = 1.0, T = 1.0, Nt = 100):
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.Nt = Nt
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))

    def simulate(self, nsims=1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = self.x0
        dt = self.T/(self.Nt)
        errs = np.random.randn(self.Nt, nsims)
        for t in range(self.Nt):
            x[t + 1,:] = x[t,:] + dt * self.mu + np.sqrt(dt) * self.sigma * errs[t,:]
        return x


class GeometricBrownianMotion:
    """
    Model parameters for the environment.
    """
    def __init__(self, x0 = 100., mu = 0.05, sigma = 0.1, T = 1.0, Nt = 100):
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.Nt = Nt
        self.timesteps = np.linspace(0, self.T, num = (Nt+1))

    def simulate(self, nsims=1):
        x = np.zeros((self.Nt+1, nsims))
        x[0,:] = self.x0
        dt = self.T/(self.Nt)
        errs = np.random.randn(self.Nt, nsims)
        for t in range(self.Nt):
            x[t + 1,:] = x[t,:] * np.exp( dt * (self.mu - 0.5*self.sigma**2) + np.sqrt(dt) * self.sigma * errs[t,:] ) 
        return x


    
def computeBlackScholesCallPrice(t,T,S,r,sigma,K):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*(T-t))/(sigma * np.sqrt(T-t))
    d2 = d1 - sigma* np.sqrt(T-t)
    return S*stat.norm.cdf(d1) - K*np.exp(-r*(T-t))*stat.norm.cdf(d2)

def computeBlackScholesPutPrice(t,T,S,r,sigma,K):
    return computeBlackScholesCallPrice(t,T,S,r,sigma,K) - (S - K*np.exp(-r*(T-t)))

def call_option_payoff(K, S):
    return np.maximum(S-K,0)

def put_option_payoff(K, S):
    return np.maximum(K-S,0)

def computeDeltaCall(t,T,S,r,sigma,K):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*(T-t))/(sigma * np.sqrt(T-t))
    return stat.norm.cdf(d1)
