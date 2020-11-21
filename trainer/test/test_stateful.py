from mlkit.start import *
from mlkit.trainer import *

class B(Stateful):
    def __init__(self):
        super().__init__()
        self._state['b'] = 20

class A(Stateful):
    def __init__(self):
        super().__init__()
        self._state['a'] = 10
        self.b = B()


a = A()
print(a.is_state_empty())
print(a.b.is_state_empty())
print(a.get_state())
a.load_state({'self': {'a': 11}, 'b': {'self': {'b': 21}}})
print(a.a)
print(a.b.b)


class C(Stateful):
    def __init__(self):
        super().__init__()

c = C()
print(c.is_state_empty())