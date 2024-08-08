import numpy as np
import pandas as pd
import logging


def estimate_propensities(T, C):
    # estimate treatment distribution for each strata of the confound 
    # directly from the data
    df = pd.DataFrame(zip(C, T), columns=['C', 'T'])
    T_levels = set(T)
    propensities = {}
    for c_level in set(C):
        subset = df.loc[df.C == c_level]
        # NOTE: subset.T => transpose
        p_TgivenC = [
            float(len(subset.loc[subset['T'] == t])) / len(subset) 
            for t in T_levels
        ]
        # propensities[c_level] = p_TgivenC[1]
        # propagate
        df.loc[df['C']==c_level,'propensity0'] = p_TgivenC[0]
        df.loc[df['C']==c_level,'propensity1'] = p_TgivenC[1]
    
    assert df['propensity0'].isnull().sum() == 0

    return df[['propensity0','propensity1']].values


def ITE_IPW(Y, outcome, propensity, treated, pair_id):
    """
    compute the effect of changing from control (rand comment) to treated (top comment) on the probability of different classes

    Params:
        Y: ground truth outcome, binary, N*1
        outcome: estimated potential outcome, prob, N * 1
        propensity: estimated propensity, prob, N * 2
        treated: binary, N * 1
        pair_id: observed and transformed counterfactual data will have the same pair id

    Return:
        effects: N * 1
    """
    logging.info(f"ITE IPW size - Y: {Y.shape}, outcome: {outcome.shape}, propensity: {propensity.shape}, binary treated var: {treated.shape}, pair id: {pair_id.shape}")
    
    overlap = np.intersect1d(pair_id[treated==1],pair_id[treated==0])
    idx1 = np.where(np.in1d(pair_id[treated==1],overlap))[0]
    idx0 = np.where(np.in1d(pair_id[treated==0],overlap))[0]
    logging.info(f"idx1 shape={idx1.shape}, idx0 {idx0.shape}")

    Y = Y[treated==0][idx0] # ground truth for observed data T=0
    p1 = propensity[treated==0,1][idx0] # probability of getting T=1 (being transformed) for observed data
    p0 = propensity[treated==0,0][idx0] # probability of getting T=0 for observed data
    o1 = outcome[treated==1][idx1] # predicted outcome for unobserved data
    o0 = outcome[treated==0][idx0] # predicted outcome for observed data

    logging.info(f"Y: {Y.shape} p1: {p1.shape}, p0: {p0.shape}, o1: {o1.shape}, o0: {o0.shape}")

    ## TODO: questionable??? makes sense???
    effects = np.multiply(1.0/p1, o1) - np.multiply(1.0/p0, o0) # N * 1
    
    logging.info(f"ITE IPW effects shape: {effects.shape}")
    
    return effects


def ITE_doubly_robust(Y, outcome, propensity, treated, pair_id):
    """
    compute the effect of changing from control (rand comment) to treated (top comment) on the probability of different classes

    Params:
        Y: ground truth outcome, binary, N*1
        outcome: estimated potential outcome, prob, N * 1
        propensity: estimated propensity, prob, N * 2
        treated: binary, N * 1
        pair_id: observed and transformed counterfactual data will have the same pair id

    Return:
        effects: N * 1
    """
    logging.info(f"ITE doubly robust size - Y: {Y.shape}, outcome: {outcome.shape}, propensity: {propensity.shape}, binary treated var: {treated.shape}, pair id: {pair_id.shape}")
    
    overlap = np.intersect1d(pair_id[treated==1],pair_id[treated==0])
    idx1 = np.where(np.in1d(pair_id[treated==1],overlap))[0]
    idx0 = np.where(np.in1d(pair_id[treated==0],overlap))[0]
    logging.info(f"idx1 shape={idx1.shape}, idx0 {idx0.shape}")

    Y = Y[treated==0][idx0] # ground truth for observed data T=0
    p1 = propensity[treated==0,1][idx0] # probability of getting T=1 (being transformed) for observed data
    p0 = propensity[treated==0,0][idx0] # probability of getting T=0 for observed data
    o1 = outcome[treated==1][idx1] # predicted outcome for unobserved data
    o0 = outcome[treated==0][idx0] # predicted outcome for observed data

    logging.info(f"Y: {Y.shape} p1: {p1.shape}, p0: {p0.shape}, o1: {o1.shape}, o0: {o0.shape}")

    # effects = np.multiply(((p1[:,1]+(1-p1[:,0]))/(p0[:,0]+(1-p0[:,1])) - 1).reshape(-1,1), (Y - o1)) + (o1-o0)
    effects = np.multiply((p1/p0 - 1), (Y - o0)) + (o1-o0) # N * 1
    
    logging.info(f"ITE DR effects shape: {effects.shape}")
    
    return effects


def CATE_IPW(Y, outcome, propensity, treated, covariates):
    """
    compute the effect of changing from control (rand comment) to treated (top comment) on the probability of different classes

    Params:
        Y: ground truth outcome, binary, N*1
        outcome: estimated potential outcome, prob, N * 1
        propensity: estimated propensity, prob, N * 2
        treated: binary, N * 1
        covariates: vector, could be one-hot encoding

    Return:
        effects: N * 1
    """
    logging.info(f"CATE IPW size - Y: {Y.shape}, outcome: {outcome.shape}, propensity: {propensity.shape}, binary treated var: {treated.shape}, covariates: {covariates.shape}")
    
    cate = {}
    if len(covariates.shape) == 1:
        covariates = covariates.reshape((-1,1))
    groups = np.unique(covariates,axis=1)
    for g in groups:
        # for each heterogeneous group
        g_mask = np.all(covariates==g, axis=1)
        g_Y = Y[g_mask]
        g_outcome = outcome[g_mask]
        g_propensity = propensity[g_mask]
        g_treated = treated[g_mask]

        y = g_Y[g_treated==0] # observed ground truth T=0
        o0 = g_outcome[g_treated==0] # predicted outcome for observed data
        o1 = g_outcome[g_treated==1] # predicted outcome for unobserved data
        p0 = g_propensity[g_treated==0,0] # probability of getting T=1 for observed data
        p1 = g_propensity[g_treated==1,1] # probability of getting T=1 for unobserved data
        # TODO: p1 and p0 NOT SAME LENGTH!!

        # logging.info(f"group {tuple(np.where(g==1)[0])} - Y: {Y.shape} p1: {p1.shape}, p0: {p0.shape}, o1: {o1.shape}, o0: {o0.shape}")

        if len(y) == 0:
            logging.info(f"group {tuple(np.where(g==1)[0])} has no data")
            continue
        cate[tuple(np.where(g==1)[0])] = np.mean(np.multiply(1.0/p1, o1)) - np.mean(np.multiply(1.0/p0, o0))
    
    logging.info(f"CATE IPW\n{cate}")

    return cate


def CATE_doubly_robust(Y, outcome, propensity, treated, covariates):
    """
    compute the effect of changing from control (rand comment) to treated (top comment) on the probability of different classes

    Params:
        Y: ground truth outcome, binary, N*1
        outcome: estimated potential outcome, prob, N * 1
        propensity: estimated propensity, prob, N * 2
        treated: binary, N * 1
        covariates: vector, could be one-hot encoding

    Return:
        effects: N * 1
    """
    logging.info(f"CATE doubly robust size - Y: {Y.shape}, outcome: {outcome.shape}, propensity: {propensity.shape}, binary treated var: {treated.shape}, covariates: {covariates.shape}")
    
    cate = {}
    if len(covariates.shape) == 1:
        covariates = covariates.reshape((-1,1))
    groups = np.unique(covariates,axis=1)
    for g in groups:
        # for each heterogeneous group
        g_mask = np.all(covariates==g, axis=1)
        g_Y = Y[g_mask]
        g_outcome = outcome[g_mask]
        g_propensity = propensity[g_mask]
        g_treated = treated[g_mask]

        y = g_Y[g_treated==0] # observed ground truth T=0
        o0 = g_outcome[g_treated==0] # predicted outcome for observed data
        o1 = g_outcome[g_treated==1] # predicted outcome for unobserved data
        p0 = g_propensity[g_treated==0,0] # probability of getting T=1 for observed data
        p1 = g_propensity[g_treated==0,1] # probability of getting T=1 for unobserved data
        # TODO: p1 and p0 NOT SAME LENGTH!!

        # logging.info(f"group {tuple(np.where(g==1)[0])} - Y: {Y.shape} p1: {p1.shape}, p0: {p0.shape}, o1: {o1.shape}, o0: {o0.shape}")

        if len(y) == 0:
            logging.info(f"group {tuple(np.where(g==1)[0])} has no data")
            continue
        cate[tuple(np.where(g==1)[0])] = np.mean(np.multiply((p1/p0 - 1), (y - o0))) + (np.mean(o1) - np.mean(o0))
    
    logging.info(f"CATE doubly robust\n{cate}")

    return cate


def CATE_naive(outcome, treated, covariates):
    logging.info(f"CATE naive size - outcome: {outcome.shape}, binary treated var: {treated.shape}, covariates: {covariates.shape}")
    
    cate = {}

    if len(covariates.shape) == 1:
        covariates = covariates.reshape((-1,1))
    groups = np.unique(covariates,axis=0)
    print(groups.shape)
    for g in groups:
        # for each heterogeneous group
        g_mask = np.all(covariates==g, axis=1)
        g_outcome = outcome[g_mask]
        g_treated = treated[g_mask]

        o0 = g_outcome[g_treated==0] # predicted outcome for observed data
        o1 = g_outcome[g_treated==1] # predicted outcome for unobserved data

        # logging.info(f"group {tuple(np.where(g==1)[0])} - o1: {o1.shape}, o0: {o0.shape}")

        if len(o0) == 0:
            logging.info(f"group {tuple(np.where(g==1)[0])} has no data")
            continue
        cate[tuple(np.where(g==1)[0])] = np.mean(o1) - np.mean(o0)

    logging.info(f"CATE naive\n{cate}")

    return cate

def ITE_naive(outcome, treated, pair_id):
    logging.info(f"ITE naive size - outcome: {outcome.shape}, binary treated var: {treated.shape}, pair id: {pair_id.shape}")
    logging.info(f"np.argsort(pair_id[treated==0]) {np.argsort(pair_id[treated==0]).shape}, np.argsort(pair_id[treated==1]) {np.argsort(pair_id[treated==1]).shape}")
    
    overlap = np.intersect1d(pair_id[treated==1],pair_id[treated==0])
    idx1 = np.where(np.in1d(pair_id[treated==1],overlap))[0]
    idx0 = np.where(np.in1d(pair_id[treated==0],overlap))[0]
    logging.info(f"idx1 shape={idx1.shape}, idx0 {idx0.shape}")
    o1 = outcome[treated==1][idx1] # predicted outcome for unobserved data
    o0 = outcome[treated==0][idx0]

    logging.info(f"o1: {o1.shape}, o0: {o0.shape}")

    effects = o1 - o0

    logging.info(f"effects shape: {effects.shape}")
    
    return effects

"""
https://www.stat.berkeley.edu/~jsteinhardt/stat260/notes/lect21.pdf
"""