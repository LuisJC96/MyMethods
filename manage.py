#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import pandas as pd
import scipy.integrate
import sys
import json
import csv
import math
from flask import Flask, request as req, Response as res, render_template, redirect, url_for
app = Flask(__name__, static_url_path='')


# # Get ChiSquare and Kolmogorov tables

# In[2]:


ChiSquare = []
with open('./resources/ChiSquare.csv', 'r' ) as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        ChiSquare.append(row)
csvFile.close()


# In[3]:


Kolmogorov = []
with open('./resources/Kolmogorov.csv', 'r' ) as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        Kolmogorov.append(row)
csvFile.close()


# # Routes

# ## Root (with html templates)

# In[4]:


@app.route('/')
def root():
    return render_template('index.html')

@app.route('/generated')
def generated():
    return redirect(url_for('root'))

@app.route('/goodness')
def goodness():
    return redirect(url_for('root'))


# ## Random generated numbers

# In[5]:


@app.route('/generated/midSquares')
def getMidSquares():
    x0 = int(req.args.get('seed'))
    reps = int(req.args.get('reps'))
    return res(json.dumps(midSquares(x0, reps)), mimetype='application/json')


# In[6]:


@app.route('/generated/linearCongruential')
def getLinearCongruential():
    x0 = int(req.args.get('seed'))
    a = int(req.args.get('mult'))
    c = int(req.args.get('inc'))
    m = int(req.args.get('mod'))
    reps = int(req.args.get('reps'))
    return res(json.dumps(linearCongruential1(x0, a, c, m, reps)), mimetype='application/json') 


# In[7]:


@app.route('/generated/mixedCongruential')
def getMixedCongruential():
    x0 = int(req.args.get('seed'))
    a = int(req.args.get('mult'))
    c = int(req.args.get('inc'))
    m = int(req.args.get('mod'))
    reps = int(req.args.get('reps'))
    return res(json.dumps(mixedCongruential(x0, a, c, m, reps)), mimetype='application/json')


# In[8]:


@app.route('/generated/multCongruential')
def getMultCongruential():
    x0 = int(req.args.get('seed'))
    a = int(req.args.get('mult'))
    m = int(req.args.get('mod'))
    reps = int(req.args.get('reps'))
    return res(json.dumps(multCongruential(x0, a, m, reps)), mimetype='application/json')


# In[9]:


@app.route('/generated/combCongruential')
def getCombCongruential():
    k = int(req.args.get('gens'))
    x0 = np.array(req.args.getlist('seed'), dtype=int)
    a = np.array(req.args.getlist('mult'), dtype=int)
    m = np.array(req.args.getlist('mod'), dtype=int)
    print(m)
    reps = int(req.args.get('reps'))
    return res(json.dumps(combCongruential(x0, a, m, k, reps)), mimetype='application/json')


# ## Random generated numbers with goodness tests

# In[10]:


@app.route('/goodness/midSquares')
def goodnessMidSquares():
    x0 = int(req.args.get('seed'))
    reps = int(req.args.get('reps'))    
    acceptance = float(req.args.get('acceptance'))
    
    gen = midSquares(x0, reps)
    chi = chiSquareGoodnessTest(gen["Randoms"], acceptance)
    kolmogorov = kolmogorovGoodnessTest (gen["Randoms"], acceptance)
    
    df = {
        "Randoms": gen,
        "Goodness" :{
            "ChiSquare" : chi,
            "Kolmogorov" : kolmogorov
        }
    }
    return res(json.dumps(df), mimetype='application/json')


# In[11]:


@app.route('/goodness/linearCongruential')
def goodnessLinearCongruential():
    x0 = int(req.args.get('seed'))
    a = int(req.args.get('mult'))
    c = int(req.args.get('inc'))
    m = int(req.args.get('mod'))
    reps = int(req.args.get('reps'))    
    acceptance = float(req.args.get('acceptance'))
    
    gen = linearCongruential1(x0, a, c, m, reps)
    chi = chiSquareGoodnessTest(gen["Randoms"], acceptance)
    kolmogorov = kolmogorovGoodnessTest (gen["Randoms"], acceptance)
    
    df = {
        "Randoms": gen,
        "Goodness" :{
            "ChiSquare" : chi,
            "Kolmogorov" : kolmogorov
        }
    }
    return res(json.dumps(df), mimetype='application/json')


# In[12]:


@app.route('/goodness/mixedCongruential')
def goodnessMixedCongruential():
    x0 = int(req.args.get('seed'))
    a = int(req.args.get('mult'))
    c = int(req.args.get('inc'))
    m = int(req.args.get('mod'))
    reps = int(req.args.get('reps'))    
    acceptance = float(req.args.get('acceptance'))
    
    gen = mixedCongruential(x0, a, c, m, reps)
    if "Error" in gen:
        return res(json.dumps(gen), mimetype='application/json')
    
    chi = chiSquareGoodnessTest(gen["Randoms"], acceptance)
    kolmogorov = kolmogorovGoodnessTest (gen["Randoms"], acceptance)
    
    df = {
        "Randoms": gen,
        "Goodness" :{
            "ChiSquare" : chi,
            "Kolmogorov" : kolmogorov
        }
    }
    return res(json.dumps(df), mimetype='application/json')


# In[13]:


@app.route('/goodness/multCongruential')
def goodnessMultCongruential():
    x0 = int(req.args.get('seed'))
    a = int(req.args.get('mult'))
    m = int(req.args.get('mod'))
    reps = int(req.args.get('reps'))    
    acceptance = float(req.args.get('acceptance'))
    
    gen = multCongruential(x0, a, m, reps)
    chi = chiSquareGoodnessTest(gen["Randoms"], acceptance)
    kolmogorov = kolmogorovGoodnessTest (gen["Randoms"], acceptance)
    
    df = {
        "Randoms": gen,
        "Goodness" :{
            "ChiSquare" : chi,
            "Kolmogorov" : kolmogorov
        }
    }
    return res(json.dumps(df), mimetype='application/json')


# In[14]:


@app.route('/goodness/combCongruential')
def goodnessCombCongruential():
    k = int(req.args.get('gens'))
    x0 = np.array(req.args.getlist('seed'), dtype=int)
    a = np.array(req.args.getlist('mult'), dtype=int)
    m = np.array(req.args.getlist('mod'), dtype=int)
    reps = int(req.args.get('reps'))
    acceptance = float(req.args.get('acceptance'))
    
    gen = combCongruential(x0, a, m, k, reps)
    chi = chiSquareGoodnessTest(gen["Randoms"], acceptance)
    kolmogorov = kolmogorovGoodnessTest (gen["Randoms"], acceptance)
    
    df = {
        "Randoms": gen,
        "Goodness" :{
            "ChiSquare" : chi,
            "Kolmogorov" : kolmogorov
        }
    }
    return res(json.dumps(df), mimetype='application/json')


# # Generating random numbers functions

# In[15]:


def midSquares(x0, reps):
    seed = np.array([], dtype=int)
    generator = []
    randoms = np.array([], dtype=int)
    rands = np.array([], dtype=int)
    periodFound = False

    for i in range(0,reps):
        seed = np.append(seed, x0)
        sqr = format(int(np.power(x0, 2)), '08')

        generator.append(sqr[:2] + "|" + sqr[2:6] + "|" + sqr[6:])
        generated = int(sqr[2:6])

        randoms = np.append(randoms, generated)
        x0 = generated
        rands = np.append(rands, x0 / 10000)
        for j in np.isin(seed, x0):
            if j:
                periodFound = True
                break
        if periodFound:
            break

    df = {'Seed': seed.tolist(),
         'Generator': generator,
         'Generated': randoms.tolist(),
         'Randoms': rands.tolist()}
    return df


# In[16]:


def linearCongruential1(x0, a, c, m, reps, contin=False):
    seed = np.array([], dtype=float)
    randoms = np.array([], dtype=float)
    rands = np.array([], dtype=float)
    periodFound = False

    for i in range(0, reps):
        seed = np.append(seed, x0)
        rep = linearCongruential2(x0, a, c, m)
        randoms = np.append(randoms, rep)
        x0 = rep
        rands = np.append(rands, rep / m)
        if not contin:
            for j in np.isin(seed, x0):
                if j:
                    periodFound = True
                    break
        if periodFound:
            break

    df = {'Seed': seed.tolist(),
         'Generated': randoms.tolist(),
         'Randoms': rands.tolist()}
    return df

def linearCongruential2(x0, a, c, m):
    return np.mod((a * x0 + c), m)


# In[17]:


def mixedCongruential(x0, a, c, m, reps):    
    if hulldobell(a, c, m):
        return linearCongruential1(x0, a, c, m, reps)
    else:        
        df = {'Error': "Does not meets Hull Dobell",
              'Suggested': suggest(x0, a, c, m)}
    return df
    
def hulldobell(a, c, m):
    if isRelativePrime(c, m):
        facs = prime_factors(m)
        for i in (np.mod((a-1), facs) == 0):
            if not(i):
                return False
        if np.mod(m, 4) == 0:
            if np.mod(a-1,4) == 0:
                return True
    return False

def suggest(x0, a, c, m):
    a1=a
    c1=c
    m1=m
    if(np.mod(m1,4)!=0):
        m1 -= np.mod(m1,4)
    while not isRelativePrime(c1, m1):
        c1+=1
    step2 = False
    primeFactors = prime_factors(m1)
    while not step2:
        check = (np.mod((a1-1), primeFactors) == 0)
        if( (not False in check) and np.mod(a1-1,4) == 0 ):
            step2 = True                    
        else:
            a1+=1                       
    return {'seed': int(x0),
            'mult' : int(a1),
            'inc' : int(c1),
            'mod': int(m1)}
    
def isRelativePrime(a, b):
    return np.gcd(int(a), int(b)) == 1

def isPrime(a):
    if num < 1:
        return False
    elif num == 2:
        return True
    else:
        for i in range(2, num):
            if np.mod(num, i) == 0:
                return False
        return True 
    
def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if np.mod(n, i):
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return np.array(factors)


# In[18]:


def multCongruential(x0, a, m, reps, contin=False):    
    if m > a and m > x0:
        return linearCongruential1(x0, a, 0, m, reps, contin=contin)


# In[19]:


def combCongruential(x0, a, m, k, reps):
    lc = []
    gen = []
    rands = []
    mk = m[0]
    for i in range(0, k):
        lc.append(multCongruential(x0[i], a[i], m[i], reps, True)["Randoms"])
    for i in range(0, reps):
        res = 0
        for j in range(0, k):
            res += np.mod((np.power(-1, j) * lc[j][i]), mk -1)
        gen.append(res)
        if res > 0 :
            rands.append(res / mk)
        elif res == 0:
            rands.append((mk-1)/mk)
            
    df = {'Seed': [],
          'Generated': gen,
          'Randoms': rands}
    
    cont = 1
    for rands in lc:
        df["Seed"].append(rands)
        cont+=1

    return df


# # Goodness Test functions

# ## Chi Square

# In[20]:


def chiSquareGoodnessTest(dataset, acceptance):
    function = lambda x : 1 / 1-0 if 0<=x and x<=1 else 0
    var = 1 #?
    minimum = np.amin(dataset)
    maximum = np.amax(dataset)
    n = len(dataset)
    rang = maximum - minimum
    k = 1+3.322*np.log10(n)
    size = rang / k    
    classmin = np.array([0], dtype=float)
    classmax = np.array([0], dtype=float)
    foabsolute = np.array([0] ,dtype=float)
    oirelative = np.array([0], dtype=float)
    eitheory = np.array([0], dtype=float)
    chi2 = np.array([0], dtype=float)    
    i = minimum    
    while i < maximum:
        classmin = np.append(classmin, i)
        classmax = np.append(classmax, i+size)
        freq = sum(1 for x in dataset if x >= i and x < (i+size))
        foabsolute = np.append(foabsolute, freq)
        oirelative = np.append(oirelative, freq / n)
        ei = scipy.integrate.quad(function, i, i+size)[0]
        eitheory = np.append(eitheory, ei)
        chi2 = np.append(chi2, np.power((freq / n - ei), 2) / ei)        
        i += size  
        
    sumChi2 = np.sum(chi2)
    chiTheory = getChiValue(var, acceptance)
    df = {
        "Chi ^2 calculated" : sumChi2,
        "Chi ^2 Theoretical" : chiTheory,
        "Accepted" : "True" if sumChi2<=chiTheory else "False"
    }
    return df


# In[21]:


def getChiValue(N, Significance):
    i = 0
    j = 0
    P = None
    while(len(ChiSquare)>j):
        if(str(N) == ChiSquare[j][0]):
            P = j
        j+=1
    while(len(ChiSquare[0])>i):
        if(str(Significance) == ChiSquare[0][i]):
            return float(ChiSquare[P][i])
        i+=1


# ## Kolmogorov 

# In[22]:


def kolmogorovGoodnessTest(dataset, acceptance):
    dataset = np.array(dataset)
    dataset.sort(axis=0)
    N = len(dataset)
    cumulativeFunction = lambda x : x / N
    cont = 1
    r = np.append([0], dataset)
    s = np.array([0], dtype=float)
    dp = np.array([0], dtype=float)
    dm = np.array([0], dtype=float)
    for i in dataset:
        s = np.append(s, cumulativeFunction(cont))
        dp = np.append(dp, np.abs(s[cont]-i))
        if cont == 1:
            dm = np.append(dm, np.abs(i))
        else:
            dm = np.append(dm, np.abs(i-(s[cont-1])))
        cont+=1;
        
    maxD = np.amax(np.append(np.amax(dp), np.amax(dm)))
    theoryD = getKolmogorovValue(N, acceptance)
    
    df = {
        "Maximum Distance": maxD,
        "Theory Distance" : theoryD,
        "Accepted" : "True" if maxD <= theoryD else "False"
    }
    return df


# In[23]:


def getKolmogorovValue(N, significance):
    i = 0
    j = 0
    P = None
    if(int(N)>50):
        if(significance == .001):
            return ((1.94947)/np.sqrt(N))
        elif(significance == .01):
            return (1.62762/np.sqrt(N))
        elif(significance == .02):
            return (1.51743/np.sqrt(N))
        elif(significance == .05):
            return (1.35810/np.sqrt(N))
        elif(significance == .1):
            return (1.22385/np.sqrt(N))
        elif(significance == .15):
            return (1.13795/np.sqrt(N))
        elif(significance == .2):
            return (1.07275/np.sqrt(N))
    else:
        while(len(Kolmogorov)>j):
            if(str(N) == Kolmogorov[j][0]):
                P = j
            j+=1
        while(len(Kolmogorov[0])>i):
            if(str(significance) == Kolmogorov[0][i]):
                return float(Kolmogorov[P][i])
            i+=1


# # Debug server run (just develop, not production)

# In[24]:


if len(sys.argv) == 2:
    if sys.argv[1] == 'debug':
        app.run(host='127.0.0.1', port=8080, debug=True)

