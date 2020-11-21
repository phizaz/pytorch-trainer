from trainer.start import *


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
assert not a.is_state_empty()
print(a.b.is_state_empty())
assert not a.b.is_state_empty()
print(a.get_state())
assert a.get_state() == {'self': {'a': 10}, 'b': {'self': {'b': 20}}}
a.load_state({'self': {'a': 11}, 'b': {'self': {'b': 21}}})
print(a.a)
assert a.a == 11
print(a.b.b)
assert a.b.b == 21


class C(Stateful):
    def __init__(self):
        super().__init__()


c = C()
print(c.is_state_empty())
assert c.is_state_empty()