# -*- coding: utf-8 -*-
""" AiiDa MusconvWorkChain class """
import numpy as np
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import ToContext, WorkChain, calcfunction, if_, while_
from aiida.plugins import WorkflowFactory

from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida_quantumespresso.common.types import ElectronicType, RelaxType, SpinType

from musconv.chkconv import ChkConvergence
from musconv.supcgen import ScGenerators


PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')

@calcfunction
def init_supcgen(aiida_struc, min_length):
    """An aiida calc function that initializes supercell generation"""
    p_st = aiida_struc.get_pymatgen_structure()

    # Calls the supercell (SC) generation class
    scg = ScGenerators(p_st)
    p_scst_mu, sc_mat, mu_frac_coord = scg.initialize(min_length.value)

    ad_scst = orm.StructureData(pymatgen=p_scst_mu)

    scmat_node = orm.ArrayData()
    scmat_node.set_array("sc_mat", sc_mat)

    vor_node = orm.ArrayData()
    vor_node.set_array("Voronoi_site", np.array(mu_frac_coord))

    return {"SC_struc": ad_scst, "SCmat": scmat_node, "Vor_site": vor_node}


@calcfunction
def re_init_supcgen(aiida_struc, ad_scst, vor_site):
    """An aiida calc function that re-initializes larger supercell generation"""

    p_st = aiida_struc.get_pymatgen_structure()
    p_scst = ad_scst.get_pymatgen_structure()

    mu_frac_coord = vor_site.get_array("Voronoi_site")

    # Calls the supercell (SC) generation class
    scg = ScGenerators(p_st)
    p_scst_mu, sc_mat = scg.re_initialize(p_scst, mu_frac_coord)

    ad_scst_out = orm.StructureData(pymatgen=p_scst_mu)

    scmat_node = orm.ArrayData()
    scmat_node.set_array("sc_mat", sc_mat)

    return {"SC_struc": ad_scst_out, "SCmat": scmat_node}


@calcfunction
def check_if_conv_achieved(aiida_struc, traj_out):
    """An aiida calc function that checks if a supercell is converged
    for intersitial defect calculations using SCF forces
    """

    atm_forc = traj_out.get_array("forces")[0]
    atm_forces = np.array(atm_forc)
    ase_struc = aiida_struc.get_ase()

    # Calls the check supercell convergence class
    csc = ChkConvergence(ase_struc, atm_forces)
    cond = csc.apply_first_crit()
    cond2 = csc.apply_2nd_crit()

    if cond is True and all(cond2):
        return orm.Bool(True)
    else:
        return orm.Bool(False)


def get_pseudos(aiida_struc, pseudofamily):
    """Get pseudos"""
    family = orm.load_group(pseudofamily)
    pseudos = family.get_pseudos(structure=aiida_struc)
    return pseudos


def get_kpoints(aiida_struc, k_density):
    """Get kpoints"""
    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(aiida_struc)
    kpoints.set_kpoints_mesh_from_density(k_density, force_parity=False)
    return kpoints


class MusconvWorkChain(ProtocolMixin, WorkChain):
    """WorkChain for finding converged supercell for interstitial impurity calculation"""

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)

        spec.input(
            "structure",
            valid_type=orm.StructureData,
            required=True,
            help="Input initial structure",
        )
        spec.input(
            "min_length",
            valid_type=orm.Float,
            default=lambda: None,
            required=False,
            help="The minimum length of the smallest"
            " lattice vector for the first generated supercell ",
        )
        spec.input(
            "max_iter_num",
            valid_type=orm.Int,
            default=lambda: orm.Int(4),
            required=False,
            help="Maximum number of iteration in the supercell convergence loop",
        )
        spec.input(
            "kpoints_distance",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.401),
            required=False,
            help="The minimum desired distance in 1/Å between k-points in reciprocal space.",
        )
        spec.input(
            "pseudo_family",
            valid_type=orm.Str,
            default=lambda: orm.Str("SSSP/1.2/PBE/efficiency"),
            required=False,
            help="The label of the pseudo family",
        )

        spec.expose_inputs(
            PwBaseWorkChain,
            namespace="pwscf",
            exclude=("pw.structure", "kpoints"),
        )  # use the  pw base workflow
        
        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace="relax",
            exclude=("structure"),
            namespace_options={
                'required': False, 'populate_defaults':False,
                'help': 'the preprocess relaxation step, if needed.',
            },
        )  # use the  pw relax workflow

        spec.outline(
            if_(cls.should_run_relax)(
                    cls.run_relax,
                    cls.inspect_relax
                ),
            cls.init_supcell_gen,
            cls.run_pw_scf,
            cls.inspect_run_get_forces,
            while_(cls.continue_iter)(
                cls.increment_n_by_one,
                if_(cls.iteration_num_not_exceeded)(
                    cls.get_larger_cell, cls.run_pw_scf, cls.inspect_run_get_forces
                ).else_(
                    cls.exit_max_iteration_exceeded,
                ),
            ),
            cls.set_outputs,
        )

        spec.output("Converged_supercell", valid_type=orm.StructureData, required=True)
        spec.output("Converged_SCmatrix", valid_type=orm.ArrayData, required=True)

        spec.exit_code(
            403,
            "ERROR_RELAXATION_FAILED",
            message="the PwRelaxWorkchain failed",
        )
        
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message="one of the PwBaseWorkChain subprocesses failed",
        )
        spec.exit_code(
            702,
            "ERROR_NUM_CONVERGENCE_ITER_EXCEEDED",
            message="Max number of supercell convergence reached ",
        )
        spec.exit_code(
            704,
            "ERROR_FITTING_FORCES_TO_EXPONENTIAL",
            message="Error in fitting the forces to an exponential",
        )

    @classmethod
    def get_builder_from_protocol(
        cls,
        pw_code,
        structure,
        protocol=None,
        overrides=None,
        electronic_type=ElectronicType.METAL,
        spin_type=SpinType.NONE,
        relax_type=None,
        initial_magnetic_moments=None,
        options=None,
        min_length=None,
        kpoints_distance=0.401,
        pseudo_family="SSSP/1.2/PBE/efficiency",
        max_iter_num=4,
        **kwargs,
    ):
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param pw_code: the ``Code`` instance configured for the ``quantumespresso.pw`` plugin.
        :param structure: the ``StructureData`` instance to use.
        :param protocol: protocol to use, if not specified, the default will be used.
        :param overrides: optional dictionary of inputs to override the defaults of the protocol.
        :param electronic_type: indicate the electronic character of the system through ``ElectronicType`` instance.
        :param spin_type: indicate the spin polarization type to use through a ``SpinType`` instance.
        :param initial_magnetic_moments: optional dictionary that maps the initial magnetic moment of each kind to a
            desired value for a spin polarized calculation. Note that in case the ``starting_magnetization`` is also
            provided in the ``overrides``, this takes precedence over the values provided here. In case neither is
            provided and ``spin_type == SpinType.COLLINEAR``, an initial guess for the magnetic moments is used.
        :param options: A dictionary of options that will be recursively set for the ``metadata.options`` input of all
            the ``CalcJobs`` that are nested in this work chain.
        :param min_length: The minimum length of the smallest lattice vector for the first generated supercell.
        :param kpoints_distance: the minimum desired distance in 1/Å between k-points in reciprocal space.
        :param pseudo_family: the label of the pseudo family.
        :param max_iter_num: Maximum number of iteration in the supercell convergence loop.
        :return: a process builder instance with all inputs defined ready for launch.
        """
        
        from aiida_quantumespresso.workflows.protocols.utils import get_starting_magnetization, recursive_merge
    
    
        if not overrides: overrides = {}
            
        overrides_pwscf = overrides.pop('pwscf',{})
        
        overrides_pwscf = recursive_merge(
                overrides_pwscf, {"CONTROL": {"tstress": True, "tprnfor": True}}
            )
        
        builder_pwscf = PwBaseWorkChain.get_builder_from_protocol(
                pw_code,
                structure,
                protocol=protocol,
                overrides=overrides_pwscf,
                electronic_type=electronic_type,
                spin_type=spin_type,
                initial_magnetic_moments=initial_magnetic_moments,
                pseudo_family=pseudo_family,
                **kwargs,
                )
        
        builder_pwscf['pw'].pop('structure', None)
        builder_pwscf.pop('kpoints_distance', None)       
        
        builder = cls.get_builder()
        
        #we can set this also wrt to some protocol, TOBE discussed
        builder.min_length=orm.Float(min_length)
        builder.kpoints_distance=orm.Float(kpoints_distance)
        builder.max_iter_num=orm.Int(max_iter_num)
        
        builder.pwscf = builder_pwscf
        
        builder.structure = structure
        builder.pseudo_family = orm.Str(pseudo_family)
        
        if relax_type:
            if relax_type != RelaxType.POSITIONS:
                raise ValueError(f'The only accepted relax_type parameter is "RelaxType.POSITIONS". You selected "{relax_type}", which is currently forbidden.')
            builder_relax = PwRelaxWorkChain.get_builder_from_protocol(
                    pw_code,
                    structure,
                    protocol=protocol,
                    overrides=overrides_pwscf,
                    electronic_type=electronic_type,
                    spin_type=spin_type,
                    initial_magnetic_moments=initial_magnetic_moments,
                    pseudo_family=pseudo_family,
                    relax_type=relax_type,
                    **kwargs,
                    )
            
            builder_relax.pop('structure', None)
            builder_relax.pop('base_final_scf', None)
            builder.relax = builder_relax
        else:
            builder.pop('relax', None)
        
        return builder
    
    def should_run_relax(self):
        return "relax" in self.inputs
    
    def run_relax(self):
        """Run the `PwBaseWorkChain` to run a relax `PwCalculation`."""

        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        inputs.structure = self.inputs.structure
        
        inputs.metadata.call_link_label = f'relax_step'

        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report(f'launching PwRelaxWorkChain<{running.pk}>')

        return ToContext(calculation_run=running)
    
    def inspect_relax(self):
        calculation = self.ctx.calculation_run
        if not calculation.is_finished_ok:
            self.report(
                f"PwRelaxWorkChain<{calculation.pk}> failed"
                "with exit status {calculation.exit_status}"
            )
            return self.exit_codes.ERROR_RELAXATION_FAILED
        else:
            self.ctx.structure = calculation.outputs.output_structure
            
        return
        
    def init_supcell_gen(self):
        """initialize supercell generation"""
        self.ctx.n = orm.Int(0)
        
        if not "structure" in self.ctx: self.ctx.structure = self.inputs.structure

        if self.inputs.min_length is None:
            m_l = min(self.ctx.structure.get_pymatgen_structure().lattice.abc) + 1
            self.inputs.min_length = orm.Float(m_l)

        result_ini = init_supcgen(self.ctx.structure, self.inputs.min_length)

        self.ctx.sup_struc_mu = result_ini["SC_struc"]
        self.ctx.musite = result_ini["Vor_site"]
        self.ctx.sc_mat = result_ini["SCmat"]

    def run_pw_scf(self):
        """Input Qe-pw structure and run pw"""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace="pwscf"))

        inputs.pw.structure = self.ctx.sup_struc_mu
        inputs.pw.pseudos = get_pseudos(
            self.ctx.sup_struc_mu, self.inputs.pseudo_family.value
        )
        inputs.kpoints = get_kpoints(
            self.ctx.sup_struc_mu, self.inputs.kpoints_distance.value
        )

        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f"running SCF calculation {running.pk}")

        return ToContext(calculation_run=running)

    def inspect_run_get_forces(self):
        """Inspect pw run and get forces"""
        calculation = self.ctx.calculation_run

        if not calculation.is_finished_ok:
            self.report(
                f"PwBaseWorkChain<{calculation.pk}> failed"
                "with exit status {calculation.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF
        else:
            self.ctx.traj_out = calculation.outputs.output_trajectory

    def continue_iter(self):
        """check convergence and decide if to continue the loop"""
        try:
            conv_res = check_if_conv_achieved(self.ctx.sup_struc_mu, self.ctx.traj_out)
            return conv_res.value == False
        except:
            self.report(
                f"Exiting MusconvWorkChain,Error in fitting the forces of supercell,"
                "iteration no. <{self.ctx.n}>) to an exponential, maybe force data not exponential"
            )
            return self.exit_codes.ERROR_FITTING_FORCES_TO_EXPONENTIAL

    def increment_n_by_one(self):
        """Increase count by 1"""
        self.ctx.n += 1

    def iteration_num_not_exceeded(self):
        """Check if iteration is exceeded"""
        return self.ctx.n <= self.inputs.max_iter_num.value

    def get_larger_cell(self):
        """Previous supercell not converged - get larger supercell"""
        result_reini = re_init_supcgen(
            self.ctx.structure, self.ctx.sup_struc_mu, self.ctx.musite
        )

        self.ctx.sup_struc_mu = result_reini["SC_struc"]
        self.ctx.sc_mat = result_reini["SCmat"]

    def exit_max_iteration_exceeded(self):
        """Exit code if max iteration number is reached"""
        self.report(
            f"Exiting MusconvWorkChain, Coverged supercell NOT achieved, next iter num"
            " <{self.ctx.n}> is greater than max iteration number {self.inputs.max_iter_num.value}"
        )
        return self.exit_codes.ERROR_NUM_CONVERGENCE_ITER_EXCEEDED

    def set_outputs(self):
        """Print outputs"""
        self.report("Setting Outputs")
        self.out("Converged_supercell", self.ctx.sup_struc_mu)
        self.out("Converged_SCmatrix", self.ctx.sc_mat)
