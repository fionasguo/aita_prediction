import numpy as np
import logging


def doubly_robust(Y, outcome, propensity, treated):
    """
    compute the effect of changing from control (rand comment) to treated (top comment) on the probability of different aita verdits (4 classes)

    Params:
        Y: observed outcome (when having top comments), one-hot, N*4
        outcome: estimated potential outcome, prob, 2N * 4 (half are treated half are control)
        propensity: estimated propensity, prob, 2N * 2 (half are treated half are control)
        treated: binary, 2N * 1

    Return:
        effects: N * 4
    """
    logging.info(f"doubly robust size - Y: {Y.shape}, outcome: {outcome.shape}, propensity: {propensity.shape}, binary treated var: {treated.shape}")
    p1 = propensity[treated==1]
    p0 = propensity[treated==0]

    o1 = outcome[treated==1]
    o0 = outcome[treated==0]

    logging.info(f"doubly robust size - p1: {p1.shape}, p0: {p0.shape}, o1: {o1.shape}, o0: {o0.shape}")

    effects = np.multiply(((p1[:,1]+(1-p1[:,0]))/(p0[:,0]+(1-p0[:,1])) - 1).reshape(-1,1), (Y - o1)) + (o1-o0)
    logging.info(f"effects shape: {effects.shape}")
    
    return effects