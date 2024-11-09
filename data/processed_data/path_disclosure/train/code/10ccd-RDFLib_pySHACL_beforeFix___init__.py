# -*- coding: utf-8 -*-

<vul/>from .value_constraints import <vul/>ClassConstraintComponent</vul></vul>

ALL_CONSTRAINT_COMPONENTS = [
    ClassConstraintComponent
]

CONSTRAINT_PARAMETERS_MAP = {p: c for c in ALL_CONSTRAINT_COMPONENTS
                             for p in c.constraint_parameters()}

ALL_CONSTRAINT_PARAMETERS = list(CONSTRAINT_PARAMETERS_MAP.keys())
