import numpy as np
from numpy import random
from architectures import *
import sys

if len(sys.argv) < 3:
    print("provide argument #1 determining nr of splits and #2 determining the dataset part # is cpu/cuda")

# fix the number of samples
N = 200
# number of runs for mean/std computation
Runs = 10
EPOCH = 50000


THR = 2.5e-03
NUM_ZEROS_THR = 1e-08
device = "cuda"

modeln = sys.argv[3]

model_fn = lambda d0, d1, d2: eval(f"{modeln}(d0, d1, d2)").to(device=device)
optimizer_fn = lambda model: torch.optim.SGD(model.parameters(), lr, momentum=0.9)
lr = 0.15

d2 = 1

if len(sys.argv) < 2:
    print("provide argument #1 determining nr of splits and #2 determining the dataset part")
splits = int(sys.argv[1])
set = int(sys.argv[2])
if set > splits - 1:
    print(f"set {set} > {splits - 1} (splits-1)")

totals = 60
samples = int(60 / splits)

if set == 0:
    starti = 2
    endi = samples
elif set == splits - 1:
    starti = set * samples
    endi = totals + 1
else:
    starti = set * samples
    endi = starti + samples
print(f"set={set}, starti={starti}, endi={endi}, samples={samples}")
d0s = list(range(starti, endi, 2))
d1s = list(range(2, 61, 2))

results = np.zeros((61, 61))
avg_losses = np.zeros((61, 61))
avg_corners = np.zeros((61, 61))

for d0 in d0s:
    for d1 in d1s:
        lossfn = torch.nn.MSELoss()

        X = 2.0 * random.rand(N, d0) - 1.0
        X = X / np.sqrt(np.sum(X**2, axis=1))[:, None]
        X = torch.tensor(X).float().to(device=device)
        print(X[0])

        Y = 2.0 * random.rand(N, d2) - 1.0
        Y = Y / np.sqrt(np.sum(Y**2, axis=1))[:, None]
        Y = torch.tensor(Y).float().to(device=device)
        print(Y[0])
        avg_corner = 0
        successes = 0
        for r in range(Runs):
            print(f"NN configuration d0={d0}, d1={d1}, d2={d2}, run={r}")
            model = model_fn(d0, d1, d2)
            optimizer = optimizer_fn(model)
            grad = []
            losse = []
            corners_cnt = []
            avg_loss = 0
            # use the full batch of data
            for e in range(EPOCH - 1):

                Ypred = model(X)
                corners_cnt.append(model.preH[np.abs(model.preH.cpu().detach().numpy()) < NUM_ZEROS_THR].shape[0])
                loss = lossfn(Ypred, Y)
                optimizer.zero_grad()
                loss.backward()
                g = 0
                for n, p in enumerate(model.parameters()):
                    g += p.grad.cpu().detach().data.norm(2)
                grad.append(g)
                optimizer.step()
                losse.append(loss.item() / np.sqrt(N))
                if e % 1000 == 0:
                    print(
                        f"set {set}, model d0={d0}, d1={d1}, d2={d2}, run={r} epoch {e} loss={loss.item() / np.sqrt(N)}, grad norm={grad[-1]}, corners_cnt={corners_cnt[-1]}"
                    )

            if losse[-1] < THR:
                successes += 1
            avg_loss += losse[-1]
            avg_corner += corners_cnt[-1]

        avg_loss /= Runs
        avg_corner /= Runs
        results[d0, d1] = successes / Runs
        avg_losses[d0, d1] = avg_loss
        avg_corners[d0, d1] = avg_corner
        print(results)

        np.savetxt(f"{modeln}_results_thr_{THR}_set{set}.csv", results, delimiter=",")
        np.savetxt(f"{modeln}_avg_final_loss_set{set}.csv", avg_losses, delimiter=",")
        np.savetxt(f"{modeln}_avg_corners_set{set}.csv", avg_corners, delimiter=",")
