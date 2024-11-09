
"""
https://www.w3.org/TR/shacl/#core-components-value-type
"""
import abc
<vul/>import pyshacl.consts</vul>

class ConstraintComponent(object, metaclass=abc.ABCMeta):

    def __init__(self, shape):
        self.shape = shape

    @classmethod
    @abc.abstractmethod
    def constraint_parameters(cls):
        return NotImplementedError()

    @abc.abstractmethod
    <vul/>def evaluate(self, target_graph, value_nodes):</vul>
        return NotImplementedError()
