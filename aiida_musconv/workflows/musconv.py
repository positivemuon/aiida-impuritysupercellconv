# -*- coding: utf-8 -*-
""" AiiDa MusconvWorkChain class """
import numpy as np
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import ToContext, WorkChain, calcfunction, if_, while_
from aiida.plugins import CalculationFactory
from musconv.chkconv import ChkConvergence
from musconv.supcgen import ScGenerators


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
def check_if_conv_achieved(aiida_struc, traj_out, conv_thr):
    """An aiida calc function that checks if a supercell is converged
    for intersitial defect calculations using SCF forces
    """

    atm_forc = traj_out.get_array("forces")[0]
    atm_forces = np.array(atm_forc)
    ase_struc = aiida_struc.get_ase()

    # Calls the check supercell convergence class
    csc = ChkConvergence(
        ase_struc=ase_struc, atomic_forces=atm_forces, conv_thr=conv_thr.value
    )
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


PwCalculation = CalculationFactory("quantumespresso.pw")


class MusconvWorkChain(WorkChain):
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
            "conv_thr",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.0257),
            required=False,
            help="Force convergence thresh in eV/Ang, default is 1e-3 au or 0.0257 ev/A",
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
            default=lambda: orm.Float(0.301),
            required=False,
            help="The minimum desired distance in 1/Å between k-points in reciprocal space.",
        )
        spec.input(
            "pseudofamily",
            valid_type=orm.Str,
            default=lambda: orm.Str("SSSP/1.2/PBE/efficiency"),
            required=False,
            help="The label of the pseudo family",
        )

        spec.expose_inputs(
            PwCalculation,
            namespace="pwscf",
            exclude=("structure", "pseudos", "kpoints"),
        )  # use the  pw calcjob

        spec.outline(
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
            402,
            "ERROR_SUB_PROCESS_FAILED_SCF",
            message="one of the PwCalculation subprocesses failed",
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

    def init_supcell_gen(self):
        """initialize supercell generation"""
        self.ctx.n = orm.Int(0)

        if self.inputs.min_length is None:
            m_l = min(self.inputs.structure.get_pymatgen_structure().lattice.abc) + 1
            self.inputs.min_length = orm.Float(m_l)

        result_ini = init_supcgen(self.inputs.structure, self.inputs.min_length)

        self.ctx.sup_struc_mu = result_ini["SC_struc"]
        self.ctx.musite = result_ini["Vor_site"]
        self.ctx.sc_mat = result_ini["SCmat"]

    def run_pw_scf(self):
        """Input Qe-pw structure and run pw"""
        inputs = AttributeDict(self.exposed_inputs(PwCalculation, namespace="pwscf"))

        inputs.structure = self.ctx.sup_struc_mu
        inputs.pseudos = get_pseudos(
            self.ctx.sup_struc_mu, self.inputs.pseudofamily.value
        )
        inputs.kpoints = get_kpoints(
            self.ctx.sup_struc_mu, self.inputs.kpoints_distance.value
        )

        running = self.submit(PwCalculation, **inputs)
        self.report(f"running SCF calculation {running.pk}")

        return ToContext(calculation_run=running)

    def inspect_run_get_forces(self):
        """Inspect pw run and get forces"""
        calculation = self.ctx.calculation_run

        if not calculation.is_finished_ok:
            self.report(
                f"PwCalculation<{calculation.pk}> failed"
                "with exit status {calculation.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF
        else:
            self.ctx.traj_out = calculation.outputs.output_trajectory

    def continue_iter(self):
        """check convergence and decide if to continue the loop"""
        try:
            conv_res = check_if_conv_achieved(
                self.ctx.sup_struc_mu, self.ctx.traj_out, self.inputs.conv_thr
            )
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
            self.inputs.structure, self.ctx.sup_struc_mu, self.ctx.musite
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
