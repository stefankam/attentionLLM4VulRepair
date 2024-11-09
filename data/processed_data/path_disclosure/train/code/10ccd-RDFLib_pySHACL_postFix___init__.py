# -*- coding: utf-8 -*-

<fix/>from .value_constraints import ClassConstraintComponent, DatatypeConstraintComponent, NodeKindConstraintComponent
from .cardinality_constraints import MinCountConstraintComponent, MaxCountConstraintComponent
from .shape_based_constraints import NodeShapeComponent, PropertyShapeComponent</fix>

ALL_CONSTRAINT_COMPONENTS = [
    <fix/>ClassConstraintComponent,
    DatatypeConstraintComponent,
    NodeKindConstraintComponent,
    MinCountConstraintComponent,
    MaxCountConstraintComponent,
    NodeShapeComponent,
    PropertyShapeComponent</fix>
]

CONSTRAINT_PARAMETERS_MAP = {p: c for c in ALL_CONSTRAINT_COMPONENTS
                             for p in c.constraint_parameters()}

ALL_CONSTRAINT_PARAMETERS = list(CONSTRAINT_PARAMETERS_MAP.keys())
