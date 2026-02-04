"""
Neural network models including PINNs and policy networks.
"""

from .world_model import PIRLWorldModel, ResidualNetwork, generate_wc_trajectory, test_world_model

try:
    from .neural_cde_observer import (
        NeuralCDEObserver,
        CDEFunc,
        Encoder,
        Decoder,
        ObserverLoss,
        generate_observer_training_data,
        test_neural_cde_observer
    )
    _HAS_CDE = True
except ImportError:
    _HAS_CDE = False

__all__ = [
    'PIRLWorldModel',
    'ResidualNetwork',
    'generate_wc_trajectory',
    'test_world_model'
]

if _HAS_CDE:
    __all__.extend([
        'NeuralCDEObserver',
        'CDEFunc',
        'Encoder',
        'Decoder',
        'ObserverLoss',
        'generate_observer_training_data',
        'test_neural_cde_observer'
    ])
