W1 = rand(3,2)
W2 = rand(3,1)


x = [ 1 2]
y = 2

x_bias = [1, x]

Z1 = x_bias*W1
A1 = sigmf(Z1,[0.5 0])

A1_bias = [1, A1]
Z2 = A1_bias*W2
A2 = sigmf(Z2,[0.5 0])

diff = (y - A2)

deltak = diff.*(sigmf(Z2,[0.5 0]).*(1-sigmf(Z2,[0.5 0])))

newW2 = A1_bias'*deltak
newW2 = deltak'*A1_bias


dback = deltak*W2'
dback = dback(:,2:end)
deltah = dback.*(sigmf(Z1,[0.5 0]).*(1-sigmf(Z1,[0.5 0])))




