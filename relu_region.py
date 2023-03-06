import torch
from math import pow
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import sys
from architectures import *

device = "cpu"

# fix the number of samples
N = 200
# number of runs for mean/std computation
Runs = 5
EPOCH = 25000

# lists of parameters to check
d0s = [20]
# d0s = [10]
d1s = [N, int(pow(N, 1.25)), int(pow(N, 1.5))]
d2s = [1]

if len(sys.argv) < 2:
    print("provide argument {0,1} 0 run BasicNet, 1 run BasicNet2")

modeln = sys.argv[1]
if modeln == "BasicNet1L":
    model_fn = lambda d0, d1, d2: BasicNet1L(d0, d1, d2, device=device)
    optimizer_fn = lambda params: torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    lr = 0.15
elif modeln == "BasicNet2L":
    model_fn = lambda d0, d1, d2: BasicNet2L(d0, d1, d2, device=device)
    # optimizer_fn = lambda model : torch.optim.Adam(model.parameters(), lr)
    optimizer_fn = lambda params: torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    lr = 0.002


T = 1e-08
lossfn = torch.nn.MSELoss()

for d0 in d0s:
    for d2 in d2s:
        # prepare the dataset
        # X ze sfery o promieniu 1
        X = 2.0 * random.rand(N, d0) - 1.0
        X = X / np.sqrt(np.sum(X**2, axis=1))[:, None]
        X = torch.tensor(X).float().to(device=device)

        print(X[0])

        Y = 2.0 * random.rand(N, d2) - 1.0
        Y = Y / np.sqrt(np.sum(Y**2, axis=1))[:, None]
        Y = torch.tensor(Y).float().to(device=device)

        print(Y[0])
        for d1 in d1s:
            losses = np.zeros((EPOCH - 1, Runs))
            Hdists = np.zeros((EPOCH - 1, Runs))
            visitedHdists = np.zeros((EPOCH - 1, Runs))
            grad_norms = np.zeros((EPOCH - 1, Runs))
            Diff_norms = np.zeros((EPOCH - 1, Runs))
            Wnorms = np.zeros((EPOCH - 1, Runs))
            small_preact_cnts = np.zeros((EPOCH - 1, Runs))
            preact_norms = np.zeros((EPOCH - 1, Runs))
            for r in range(Runs):
                print(f"NN configuration d0={d0}, d1={d1}, d2={d2}, run={r}")
                model = model_fn(d0, d1, d2)
                # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
                optimizer = optimizer_fn(model)
                initW = model.W.weight.cpu().detach().numpy().copy()
                prev_grads = []

                visitedH = {}
                # compute initial H
                with torch.no_grad():
                    Ypred = model(X)
                initH = model.H > 0
                H = initH
                m = 1 * (model.H > 0).cpu().detach().numpy()
                hdist = []
                visitedHdist = []
                visitedHdict = {}
                visitedHdict[initH.cpu().detach().numpy().tobytes()] = 1
                Wnorm = []
                grad = []
                diff = []
                preact_norm = []
                losse = []
                small_preact_cnt = []
                with torch.no_grad():
                    prevLoss = lossfn(Ypred, Y)
                # use the full batch of data
                for e in range(EPOCH - 1):
                    prevH = H
                    Ypred = model(X)
                    H = model.H > 0
                    Wnorm.append(np.linalg.norm(initW - model.W.weight.cpu().detach().numpy()))
                    visitedHdist.append(np.sum(1 * prevH.bitwise_xor(H).cpu().detach().numpy()))
                    if H.cpu().detach().numpy().tobytes() not in visitedHdict.keys():
                        visitedHdict[H.cpu().detach().numpy().tobytes()] = 1
                    hdiff = 1 * initH.bitwise_xor(H).cpu().detach().numpy()
                    hdist.append(np.sum(hdiff))
                    small_preact_cnt.append(model.preH[np.abs(model.preH.cpu().detach().numpy()) < T].shape[0])
                    preact_norm.append(np.linalg.norm(model.preH.cpu().detach().numpy()))
                    loss = lossfn(Ypred, Y)
                    optimizer.zero_grad()
                    loss.backward()
                    if e == 0:
                        for n, p in enumerate(model.parameters()):
                            prev_grads.append(p.grad.cpu().detach().numpy().copy())

                    g = 0
                    gds = []
                    ds = []
                    for n, p in enumerate(model.parameters()):
                        g += p.grad.cpu().detach().data.norm(2)
                        # relative change of the differential
                        gds.append(np.linalg.norm(prev_grads[n] - p.grad.cpu().detach().numpy(), ord=2))
                        ds.append(np.linalg.norm(prev_grads[n], ord=2))
                        prev_grads[n] = p.grad.cpu().detach().numpy().copy()

                    grad.append(g)
                    diff.append(np.max(gds) / np.max(ds))
                    optimizer.step()
                    if e == 50000:
                        print("halving the optimizer lr")
                        for g in optimizer.param_groups:
                            g["lr"] = g["lr"] / 2.0

                    losse.append(np.abs(loss.item() - prevLoss) / prevLoss)
                    prevLoss = loss.item()
                    if e % 10 == 0:
                        print(
                            f"model {1} epoch {e} loss={losse[-1]}, grad norm={grad[-1]}, hdist={visitedHdist[-1]}, small_preact_cnt={small_preact_cnt[-1]}, Wdiff={Wnorm[-1]}, diff={diff[-1]}"
                        )
                losses[:, r] = np.array(losse)
                grad_norms[:, r] = np.array(grad)
                Diff_norms[:, r] = np.array(diff)
                Hdists[:, r] = np.array(hdist)
                visitedHdists[:, r] = np.array(visitedHdist)
                Wnorms[:, r] = np.array(Wnorm)
                small_preact_cnts[:, r] = np.array(small_preact_cnt)
                preact_norms[:, r] = np.array(preact_norm)
                print(f"visited {len(visitedHdict.keys())} distinct regions")

            # save data
            np.savetxt(f"{modeln}/{modeln}_d0_{d0}_d1_{d1}_d2_{d2}_losses.csv", losses, delimiter=",")
            np.savetxt(f"{modeln}/{modeln}_d0_{d0}_d1_{d1}_d2_{d2}_grad_norms.csv", grad_norms, delimiter=",")
            np.savetxt(f"{modeln}/{modeln}_d0_{d0}_d1_{d1}_d2_{d2}_Hdists.csv", Hdists, delimiter=",")
            np.savetxt(f"{modeln}/{modeln}_d0_{d0}_d1_{d1}_d2_{d2}_visitedHdists.csv", visitedHdists, delimiter=",")
            np.savetxt(f"{modeln}/{modeln}_d0_{d0}_d1_{d1}_d2_{d2}_Wnorms.csv", Wnorms, delimiter=",")
            np.savetxt(f"{modeln}/{modeln}_d0_{d0}_d1_{d1}_d2_{d2}_Diff_norms.csv", Diff_norms, delimiter=",")
            np.savetxt(
                f"{modeln}/{modeln}_d0_{d0}_d1_{d1}_d2_{d2}_small_preact_cnts.csv", small_preact_cnts, delimiter=","
            )
            np.savetxt(f"{modeln}/{modeln}_d0_{d0}_d1_{d1}_d2_{d2}_preact_norms.csv", preact_norms, delimiter=",")

# plot same example data
plt.title("Hamming distance from the initial relu activation patterns")
plt.plot(hdist)
plt.savefig("initHdist.png")
plt.show()


visitedHdist = np.array(visitedHdist)
print(np.min(visitedHdist), np.max(visitedHdist))
plt.title("Hamming distance between two consecutive relu activation patterns")
plt.plot(visitedHdist)
plt.savefig("consecHdist.png")
plt.show()

plt.title("W norms")
plt.plot(Wnorms)
plt.show()

plt.title("loss gradient norms")
plt.plot(grad)
plt.show()

plt.title("pre-activation norms")
plt.plot(preact_norm)
plt.show()

plt.title("small pre-activation counts")
plt.plot(small_preact_cnt)
plt.show()

plt.title("pre-activation norms")
plt.plot(preact_norm)
plt.show()

with torch.no_grad():
    finalPreH = model.preH

# preliminary corners investigation

print(f"small preactivations (less than {T}):")
print(finalPreH[np.abs(finalPreH.cpu().detach().numpy()) < T].cpu().detach().numpy())
nr = finalPreH[np.abs(finalPreH.cpu().detach().numpy()) < T].shape[0]
print(
    f"# of small preactivations {finalPreH[np.abs(finalPreH.cpu().detach().numpy()) < T].shape}, fraction of total {nr/(d0*d1)}"
)
