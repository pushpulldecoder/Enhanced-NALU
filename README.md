# Enhanced-NALU
Enhanced Neural Arithmetic Logic Unit

<!-- > Looking down the misty path to uncertain destinations🌌🍀&nbsp;&nbsp;&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- x' <br><br><br> -->

It is modified implementation of Neural Arithematic Logic Unit as discussed in <a href=https://arxiv.org/abs/1808.00508>this paper</a><br>



## Dependencies
- PyTorch
- Numpy
- Torchviz

## Neural Accumulator
Neural accumulator is specifically used for either addition or subtraction<br>
It learns the operation of add/sub as feed forward

>    no_input  : 2<br>
>    no_output : 1


>    <b>w_hat.shape</b>  : (1, 2) <br>
>    <b>m_hat.shape</b>  : (1, 2)

>    <b>x.shape</b>      : (n, 2)<br>
>    <b>W.shape</b>      : (1, 2)<br>

>    <b>return</b>       :z (x * W.T) + bias<br>
>    <b>return.shape</b> : (n, 1)

>    <b>sigmoid(m_hat)</b>-------converges to----\>(1, 1)<br>
sigmoid(m_hat) will converge to (1, 1) to make dot product of matrices like summation of inputs

>    <b>tanh(w_hat)</b>-------converges to----\>(1, 1)--------addition<br>
>    <b>tanh(w_hat)</b>-------converges to----\>(1, 1)--------subtraction<br>
tanh(w_hat) will either converge to (1, 1) or to (1, -1) depending if the operation is addition or subtraction respectily

```python
class NAC(torch.nn.Module):
    
    def __init__(self, parameter):
        super().__init__()
        self.no_inp = parameter.get("no_input")
        self.no_out = parameter.get("no_output")
        self.w_hat = torch.nn.Parameter(torch.Tensor(self.no_out, self.no_inp).to(DEVICE))
        self.m_hat = torch.nn.Parameter(torch.Tensor(self.no_out, self.no_inp).to(DEVICE))
        torch.nn.init.xavier_normal_(self.w_hat)
        torch.nn.init.xavier_normal_(self.m_hat)
        self.bias = None
        
    def forward(self, x):
        W = torch.tanh(self.w_hat) * torch.sigmoid(self.m_hat)
        return torch.nn.functional.linear(x, W, self.bias)
```

## Multiplication Unit

MU with the help of NAC performs higher order operations like multiplication and division

F = sigmoid(f) will either converge to 0 or 1 giving the input of add/sub and mul/div respectively<br>
It works as gate or barrier to perform operation of add/sub or mul/div

a : Stores output of add/sub of given inputs<br>
m : Stores output of mul/div of given inputs

$m = p * q$<br>
$m = e^{log(pq)}$<br>
$m = e^{log(p) + log(q)}$<br>
$m = e^{NAC(log(p), log(q))}$<br>

```python
m = self.nac(torch.log(torch.abs(x) + self.eps))
m = torch.exp(m)
```

## Power Unit
PU performs even higher order operations like power(incuding roots)<br>
$p = a^b$<br>
$p = e^{log(a^b)}$<br>
$p = e^{b*log(a)}$<br>
$p = e^{MLU(b, log(a))}$<br>

```python
p = self.mu(torch.stack((torch.log(torch.abs(x[:, 0])), x[:, 1])).T)
p = torch.exp(p)
```

## Generating Data

For all basic mathematical operations, we are sampling from numbers from uniform distribution

```python
def data_generator(min_val, max_val, num_obs, op):
    data = np.random.uniform(min_val, max_val, size=(num_obs, 2))
    if op == '+':
        targets = data[:, 0] + data[:, 1]
    elif op == '-':
        targets = data[:, 0] - data[:, 1]
    elif op == '*':
        targets = data[:, 0] * data[:, 1]
    elif op == '/':
        targets = data[:, 0] / data[:, 1]
    elif op == 'p':
        targets = np.power(data[:, 0], data[:, 1])
    return data, targets
```


<br>
<br>

