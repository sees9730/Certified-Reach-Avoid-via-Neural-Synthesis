"""Differentiable float64 interval proposals; proof uses mpmath separately."""
import math
import torch


class I:
    def __init__(self, lo, hi=None):
        self.lo = torch.as_tensor(lo, dtype=torch.float64)
        self.hi = self.lo if hi is None else torch.as_tensor(hi, dtype=torch.float64)

    def __add__(self, other):
        other = wrap(other)
        return I(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self): return I(-self.hi, -self.lo)
    def __sub__(self, other): return self + -wrap(other)
    def __rsub__(self, other): return wrap(other) + -self

    def __mul__(self, other):
        other = wrap(other)
        values = torch.stack(torch.broadcast_tensors(self.lo*other.lo, self.lo*other.hi,
                                                     self.hi*other.lo, self.hi*other.hi))
        return I(values.amin(0), values.amax(0))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = wrap(other)
        if torch.any((other.lo <= 0) & (other.hi >= 0)):
            raise ValueError("Interval division through zero")
        return self * I(1/other.hi, 1/other.lo)

    def __getitem__(self, item): return I(self.lo[item], self.hi[item])
    def sum(self, dim): return I(self.lo.sum(dim), self.hi.sum(dim))
    def exp(self): return I(self.lo.exp(), self.hi.exp())
    def tanh(self): return I(self.lo.tanh(), self.hi.tanh())
    def sigmoid(self): return I(self.lo.sigmoid(), self.hi.sigmoid())

    def square(self):
        low = torch.minimum(self.lo.square(), self.hi.square())
        return I(torch.where((self.lo <= 0) & (self.hi >= 0), 0., low),
                 torch.maximum(self.lo.square(), self.hi.square()))

    def sin(self):
        low, high = torch.minimum(self.lo.sin(), self.hi.sin()), torch.maximum(self.lo.sin(), self.hi.sin())
        contains = lambda phase: torch.ceil((self.lo-phase)/(2*math.pi)) <= torch.floor((self.hi-phase)/(2*math.pi))
        return I(torch.where(contains(-math.pi/2), -1., low),
                 torch.where(contains(math.pi/2), 1., high))

    def cos(self): return (self + math.pi/2).sin()


def wrap(value): return value if isinstance(value, I) else I(value)


def quadratic(x, a, b):
    """Range of a*x^2+b*x for a>0, retaining the repeated-variable dependency."""
    f = lambda t: a*t*t+b*t
    low, high = torch.minimum(f(x.lo), f(x.hi)), torch.maximum(f(x.lo), f(x.hi))
    vertex = -b/(2*a)
    return I(torch.where((x.lo <= vertex) & (x.hi >= vertex), f(vertex), low), high)
