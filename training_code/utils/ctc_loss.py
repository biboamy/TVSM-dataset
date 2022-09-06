import numpy
numpy.seterr(divide = 'ignore')
import torch
from torch.autograd import Variable
import functools

# loss function from https://github.com/MaigoAkisame/cmu-thesis/blob/master/code/sequential/ctc.py
# paper https://maigoakisame.github.io/papers/icassp17.pdf
# Input arguments:
#   frameProb: a 3-D Variable of size N_SEQS * N_FRAMES * N_CLASSES containing the probability of each event at each frame.
#   seqLen: a list or numpy array indicating the number of valid frames in each sequence.
#   label: a list of label sequences.
# Note on implementation:
#   Anything that will be backpropped must be a Variable;
#   Anything used as an index must be a torch.cuda.LongTensor.
def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

def tensor(array):
    if array.dtype == 'bool':
        array = array.astype('uint8')
    return cuda(torch.from_numpy(array))

def variable(array):
    if isinstance(array, numpy.ndarray):
        array = tensor(array)
    return cuda(Variable(array))

def logsumexp(*args):
    M = functools.reduce(torch.max, args)
    mask = M != -numpy.inf
    M[mask] += torch.log(sum(torch.exp(x[mask] - M[mask]) for x in args))
        # Must pick the valid part out, otherwise the gradient will contain NaNs
    return M

def ctl_loss(frameProb, seqLen, label, labelLen, maxConcur = 1, debug = False):
    nSeqs, nFrames, nClasses = frameProb.size()

    # Clear the content in the frames of frameProb beyond seqLen
    frameIndex = variable(torch.arange(nFrames).repeat(nSeqs, 1))
    mask = variable((frameIndex < seqLen.reshape((nSeqs, 1))).unsqueeze(2))
    z = variable(torch.zeros(frameProb.size()))
    frameProb = torch.where(mask, frameProb, z)

    # Convert frameProb (probabilities of events) into probabilities of event boundaries
    z = variable(1e-7 * torch.ones((nSeqs, 1, nClasses)))  # Real zeros would cause NaNs in the gradient
    frameProb = torch.cat([z, frameProb, z], dim=1)
    startProb = torch.clamp(frameProb[:, 1:] - frameProb[:, :-1], min=1e-7)
    endProb = torch.clamp(frameProb[:, :-1] - frameProb[:, 1:], min=1e-7)
    boundaryProb = torch.stack([startProb, endProb], dim=3).view((nSeqs, nFrames + 1, nClasses * 2))
    blankLogProb = torch.log(1 - boundaryProb).sum(dim=2)
    # blankLogProb[seq, frame] = log probability of emitting nothing at this frame
    deltaLogProb = torch.log(boundaryProb) - torch.log(1 - boundaryProb)
    # deltaLogProb[seq, frame, token] = log prob of emitting token minus log prob of not emitting token

    # Put the label sequences into a Variable
    maxLabelLen = labelLen.max()

    if maxConcur > maxLabelLen:
        maxConcur = maxLabelLen

    # Compute alpha trellis
    # alpha[m, n] = log probability of having emitted n tokens in the m-th sequence at the current frame
    nStates = maxLabelLen + 1
    alpha = variable(-numpy.inf * torch.ones((nSeqs, nStates)))
    alpha[:, 0] = 0
    seqIndex = variable(torch.arange(nSeqs).repeat(nStates, 1).T)
    dummyColumns = variable(-float('inf') * torch.ones((nSeqs, maxConcur)))
    uttLogProb = variable(torch.zeros(nSeqs))

    for frame in range(nFrames + 1):  # +1 because we are considering boundaries
        # Case 0: don't emit anything at current frame
        p = alpha + blankLogProb[:, frame].view((-1, 1))
        alpha = p

        for i in range(1, maxConcur + 1):
            # Case i: emit i tokens at current frame
            p = p[:, :-1] + deltaLogProb[seqIndex[:, i:], frame, label[:, (i - 1):]]
            alpha = logsumexp(alpha, torch.cat([dummyColumns[:, :i], p], dim=1))
        # Collect probability for ends of utterances
        finishedSeqs = (seqLen == frame).nonzero()#[0]
        if len(finishedSeqs) > 0:
            finishedSeqs = variable(finishedSeqs)
            uttLogProb[finishedSeqs] = alpha[finishedSeqs, labelLen[finishedSeqs]].clone()

    # Return the per-frame negative log probability of all utterances (and per-utterance log probs if debug == True)
    uttLogProb[uttLogProb != uttLogProb] = 0
    uttLogProb[torch.isinf(uttLogProb)] = 0
    seqLen[uttLogProb != uttLogProb] = 0
    seqLen[torch.isinf(uttLogProb)] = 0
    loss = -uttLogProb.sum() / (seqLen + 1).sum()
    if debug:
        return loss, uttLogProb
    else:
        return loss

# Input arguments:
#   logProb: a 3-D Variable of size N_SEQS * N_FRAMES * N_LABELS containing LOG probabilities.
#   seqLen: a list or numpy array indicating the number of valid frames in each sequence.
#   label: a list of label sequences.
# Note on implementation:
#   Anything that will be backpropped must be a Variable;
#   Anything used as an index must be a torch.cuda.LongTensor.
def ctc_loss(logProb, seqLen, label, labelLen, debug = False):
    nSeqs, nFrames = logProb.size(0), logProb.size(1)
    # Insert blank symbol at the beginning, at the end, and between all symbols of the label sequences
    nStates = max(len(x) for x in label) * 2 + 1
    extendedLabel = variable(torch.zeros((nSeqs, nStates), dtype=torch.int64))
    for i in range(nSeqs):
        extendedLabel[i, 1 : (len(label[i]) * 2) : 2] = label[i]
    label = extendedLabel
    # Compute alpha trellis
    dummyColumn = -numpy.inf * variable(torch.ones((nSeqs, 1)))
    allSeqIndex = variable(torch.arange(nSeqs).repeat(nStates, 1)).T
    uttLogProb = variable(torch.zeros(nSeqs))
    for frame in range(nFrames):
        if frame == 0:
            # Initialize the log probability first two states to log(1), and other states to log(0)
            alpha = (-numpy.inf * variable(torch.ones((nSeqs, nStates))))
            alpha[:, :2] = 0
        else:
            # Receive probability from previous frame
            p2 = alpha[:, :-2].clone()
            p2[label[:, 2:] == label[:, :-2]] = -numpy.inf
            # Probability can pass across labels two steps apart if they are different
            alpha = logsumexp(alpha, torch.cat([dummyColumn, alpha[:, :-1]], 1), torch.cat([dummyColumn, dummyColumn, p2], 1))
        # Multiply with the probability of current frame
        alpha += logProb[allSeqIndex, frame, label]
        # Collect probability for ends of utterances
        seqIndex = (seqLen == frame + 1).nonzero()#[0]
        if len(seqIndex) > 0:
            seqIndex = variable(seqIndex)
            ll = labelLen[seqIndex]
            p = alpha[seqIndex, ll * 2].clone()
            if (ll > 0).any():
                p[ll > 0] = logsumexp(p[ll > 0], alpha[seqIndex[ll > 0], ll[ll > 0] * 2 - 1])
            uttLogProb[seqIndex] = p

    # Return the per-frame negative log probability of all utterances (and per-utterance log probs if debug == True)
    loss = -uttLogProb.sum() / seqLen.sum()
    if debug:
        return loss, uttLogProb
    else:
        return loss