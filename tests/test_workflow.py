# -*- coding: utf-8 -*-
"""Tests for the `musConvWorkChain` class."""
import pytest
from aiida import orm
from aiida.plugins import DataFactory
from aiida_musConv.workflows.musConv import musConvWorkChain


@pytest.fixture
def generate_builder(generate_structure, fixture_code):
    """Generate default inputs for `musConvWorkChain`"""

    def _get_builder():
        """Generate default builder for `musConvWorkChain`"""

        inputstructure = generate_structure("Si")
        code = fixture_code("quantumespresso.pw")

        builder = musConvWorkChain.get_builder()
        builder.structure = inputstructure

        Dict = DataFactory("dict")
        paramters = {
            "CONTROL": {
                "calculation": "scf",
                "restart_mode": "from_scratch",
                "tstress": True,
                "tprnfor": True,
            },
            "SYSTEM": {
                "ecutwfc": 30.0,
                "ecutrho": 240.0,
                "tot_charge": 1.0,
                "occupations": "smearing",
                "smearing": "cold",
                "degauss": 0.01,
            },
            "ELECTRONS": {
                "conv_thr": 1.0e-6,
                "electron_maxstep": 300,
                "mixing_beta": 0.3,
            },
        }

        builder.pwscf.parameters = Dict(dict=paramters)
        builder.pwscf.metadata.options.resources = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }

        return builder

    return _get_builder


@pytest.fixture
def generate_workchain(generate_builder):
    """Generate an instance of musConvWorkChain"""

    def _generate_workchain(exit_code=None):
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager

        builder = generate_builder()
        runner = get_manager().get_runner()
        process = instantiate_process(runner, builder)

        if exit_code is not None:
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

        return process

    return _generate_workchain


def test_initialize(aiida_profile, generate_workchain):
    """
    Test `musConvWorkChain.initialization`.
    This checks that we can create the workchain successfully, and that it is initialised into the correct state.
    """
    process = generate_workchain()
    assert process.init_supcell_gen() is None
    assert process.ctx.n.value == 0
    # assert process.ctx.n == 1
    assert isinstance(process.ctx.sup_struc_mu, orm.StructureData)
    assert isinstance(process.ctx.musite, orm.ArrayData)
    assert isinstance(process.ctx.sc_matrix, orm.ArrayData)
