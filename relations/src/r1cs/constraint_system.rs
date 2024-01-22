#[cfg(feature = "std")]
use crate::r1cs::ConstraintTrace;
use crate::r1cs::{LcIndex, LinearCombination, Matrix, SynthesisError, Variable};
use ark_ff::Field;
use ark_std::{
    any::{Any, TypeId},
    boxed::Box,
    cell::{Ref, RefCell, RefMut},
    collections::BTreeMap,
    format,
    rc::Rc,
    string::String,
    vec,
    vec::Vec,
};
use async_recursion::async_recursion;
use std::sync::Arc;
use tokio::sync::{Mutex, MutexGuard};

/// Computations are expressed in terms of rank-1 constraint systems (R1CS).
/// The `generate_constraints` method is called to generate constraints for
/// both CRS generation and for proving.
// TODO: Think: should we replace this with just a closure?
pub trait ConstraintSynthesizer<F: Field> {
    /// Drives generation of new constraints inside `cs`.
    fn generate_constraints(self, cs: ConstraintSystemRef<F>) -> crate::r1cs::Result<()>;
}

/// An Rank-One `ConstraintSystem`. Enforces constraints of the form
/// `⟨a_i, z⟩ ⋅ ⟨b_i, z⟩ = ⟨c_i, z⟩`, where `a_i`, `b_i`, and `c_i` are linear
/// combinations over variables, and `z` is the concrete assignment to these
/// variables.
#[derive(Debug, Clone)]
pub struct ConstraintSystem<F: Field> {
    /// The mode in which the constraint system is operating. `self` can either
    /// be in setup mode (i.e., `self.mode == SynthesisMode::Setup`) or in
    /// proving mode (i.e., `self.mode == SynthesisMode::Prove`). If we are
    /// in proving mode, then we have the additional option of whether or
    /// not to construct the A, B, and C matrices of the constraint system
    /// (see below).
    pub mode: SynthesisMode,
    /// The number of variables that are "public inputs" to the constraint
    /// system.
    pub num_instance_variables: usize,
    /// The number of variables that are "private inputs" to the constraint
    /// system.
    pub num_witness_variables: usize,
    /// The number of constraints in the constraint system.
    pub num_constraints: usize,
    /// The number of linear combinations
    pub num_linear_combinations: usize,

    /// The parameter we aim to minimize in this constraint system (either the
    /// number of constraints or their total weight).
    pub optimization_goal: OptimizationGoal,

    /// Assignments to the public input variables. This is empty if `self.mode
    /// == SynthesisMode::Setup`.
    pub instance_assignment: Vec<F>,
    /// Assignments to the private input variables. This is empty if `self.mode
    /// == SynthesisMode::Setup`.
    pub witness_assignment: Vec<F>,

    /// Map for gadgets to cache computation results.
    pub cache_map: Arc<Mutex<BTreeMap<TypeId, Box<dyn Any>>>>,

    lc_map: BTreeMap<LcIndex, LinearCombination<F>>,

    #[cfg(feature = "std")]
    constraint_traces: Vec<Option<ConstraintTrace>>,

    a_constraints: Vec<LcIndex>,
    b_constraints: Vec<LcIndex>,
    c_constraints: Vec<LcIndex>,

    lc_assignment_cache: Arc<Mutex<BTreeMap<LcIndex, F>>>,
}

impl<F: Field> Default for ConstraintSystem<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Defines the mode of operation of a `ConstraintSystem`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum SynthesisMode {
    /// Indicate to the `ConstraintSystem` that it should only generate
    /// constraint matrices and not populate the variable assignments.
    Setup,
    /// Indicate to the `ConstraintSystem` that it populate the variable
    /// assignments. If additionally `construct_matrices == true`, then generate
    /// the matrices as in the `Setup` case.
    Prove {
        /// If `construct_matrices == true`, then generate
        /// the matrices as in the `Setup` case.
        construct_matrices: bool,
    },
}

/// Defines the parameter to optimize for a `ConstraintSystem`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum OptimizationGoal {
    /// Make no attempt to optimize.
    None,
    /// Minimize the number of constraints.
    Constraints,
    /// Minimize the total weight of the constraints (the number of nonzero
    /// entries across all constraints).
    Weight,
}

impl<F: Field> ConstraintSystem<F> {
    #[inline]
    fn make_row(&self, l: &LinearCombination<F>) -> Vec<(F, usize)> {
        let num_input = self.num_instance_variables;
        l.0.iter()
            .filter_map(|(coeff, var)| {
                if coeff.is_zero() {
                    None
                } else {
                    Some((
                        *coeff,
                        var.get_index_unchecked(num_input).expect("no symbolic LCs"),
                    ))
                }
            })
            .collect()
    }

    /// Construct an empty `ConstraintSystem`.
    pub fn new() -> Self {
        Self {
            num_instance_variables: 1,
            num_witness_variables: 0,
            num_constraints: 0,
            num_linear_combinations: 0,
            a_constraints: Vec::new(),
            b_constraints: Vec::new(),
            c_constraints: Vec::new(),
            instance_assignment: vec![F::one()],
            witness_assignment: Vec::new(),
            cache_map: Arc::new(Mutex::new(BTreeMap::new())),
            #[cfg(feature = "std")]
            constraint_traces: Vec::new(),

            lc_map: BTreeMap::new(),
            lc_assignment_cache: Arc::new(Mutex::new(BTreeMap::new())),

            mode: SynthesisMode::Prove {
                construct_matrices: true,
            },

            optimization_goal: OptimizationGoal::Constraints,
        }
    }

    /// Create a new `ConstraintSystemRef<F>`.
    pub fn new_ref() -> ConstraintSystemRef<F> {
        ConstraintSystemRef::new(Self::new())
    }

    /// Set `self.mode` to `mode`.
    pub fn set_mode(&mut self, mode: SynthesisMode) {
        self.mode = mode;
    }

    /// Check whether `self.mode == SynthesisMode::Setup`.
    pub fn is_in_setup_mode(&self) -> bool {
        self.mode == SynthesisMode::Setup
    }

    /// Check whether this constraint system aims to optimize weight,
    /// number of constraints, or neither.
    pub fn optimization_goal(&self) -> OptimizationGoal {
        self.optimization_goal
    }

    /// Specify whether this constraint system should aim to optimize weight,
    /// number of constraints, or neither.
    pub fn set_optimization_goal(&mut self, goal: OptimizationGoal) {
        // `set_optimization_goal` should only be executed before any constraint or value is created.
        assert_eq!(self.num_instance_variables, 1);
        assert_eq!(self.num_witness_variables, 0);
        assert_eq!(self.num_constraints, 0);
        assert_eq!(self.num_linear_combinations, 0);

        self.optimization_goal = goal;
    }

    /// Check whether or not `self` will construct matrices.
    pub fn should_construct_matrices(&self) -> bool {
        match self.mode {
            SynthesisMode::Setup => true,
            SynthesisMode::Prove { construct_matrices } => construct_matrices,
        }
    }

    /// Return a variable representing the constant "zero" inside the constraint
    /// system.
    #[inline]
    pub fn zero() -> Variable {
        Variable::Zero
    }

    /// Return a variable representing the constant "one" inside the constraint
    /// system.
    #[inline]
    pub fn one() -> Variable {
        Variable::One
    }

    /// Obtain a variable representing a new public instance input.
    #[inline]
    pub fn new_input_variable<Func>(&mut self, f: Func) -> crate::r1cs::Result<Variable>
    where
        Func: FnOnce() -> crate::r1cs::Result<F>,
    {
        let index = self.num_instance_variables;
        self.num_instance_variables += 1;

        if !self.is_in_setup_mode() {
            self.instance_assignment.push(f()?);
        }
        Ok(Variable::Instance(index))
    }

    /// Obtain a variable representing a new private witness input.
    #[inline]
    pub fn new_witness_variable<Func>(&mut self, f: Func) -> crate::r1cs::Result<Variable>
    where
        Func: FnOnce() -> crate::r1cs::Result<F>,
    {
        let index = self.num_witness_variables;
        self.num_witness_variables += 1;

        if !self.is_in_setup_mode() {
            self.witness_assignment.push(f()?);
        }
        Ok(Variable::Witness(index))
    }

    /// Obtain a variable representing a linear combination.
    #[inline]
    pub fn new_lc(&mut self, lc: LinearCombination<F>) -> crate::r1cs::Result<Variable> {
        let index = LcIndex(self.num_linear_combinations);
        let var = Variable::SymbolicLc(index);

        self.lc_map.insert(index, lc);

        self.num_linear_combinations += 1;
        Ok(var)
    }

    /// Enforce a R1CS constraint with the name `name`.
    #[inline]
    pub fn enforce_constraint(
        &mut self,
        a: LinearCombination<F>,
        b: LinearCombination<F>,
        c: LinearCombination<F>,
    ) -> crate::r1cs::Result<()> {
        if self.should_construct_matrices() {
            let a_index = self.new_lc(a)?.get_lc_index().unwrap();
            let b_index = self.new_lc(b)?.get_lc_index().unwrap();
            let c_index = self.new_lc(c)?.get_lc_index().unwrap();
            self.a_constraints.push(a_index);
            self.b_constraints.push(b_index);
            self.c_constraints.push(c_index);
        }
        self.num_constraints += 1;
        #[cfg(feature = "std")]
        {
            let trace = ConstraintTrace::capture();
            self.constraint_traces.push(trace);
        }
        Ok(())
    }

    /// Count the number of times each LC is used within other LCs in the
    /// constraint system
    fn lc_num_times_used(&self, count_sinks: bool) -> Vec<usize> {
        let mut num_times_used = vec![0; self.lc_map.len()];

        // Iterate over every lc in constraint system
        for (index, lc) in self.lc_map.iter() {
            num_times_used[index.0] += count_sinks as usize;

            // Increment the counter for each lc that this lc has a direct dependency on.
            for &(_, var) in lc.iter() {
                if var.is_lc() {
                    let lc_index = var.get_lc_index().expect("should be lc");
                    num_times_used[lc_index.0] += 1;
                }
            }
        }
        num_times_used
    }

    /// Transform the map of linear combinations.
    /// Specifically, allow the creation of additional witness assignments.
    ///
    /// This method is used as a subroutine of `inline_all_lcs` and `outline_lcs`.
    ///
    /// The transformer function is given a references of this constraint system (&self),
    /// number of times used, and a mutable reference of the linear combination to be transformed.
    ///     (&ConstraintSystem<F>, usize, &mut LinearCombination<F>)
    ///
    /// The transformer function returns the number of new witness variables needed
    /// and a vector of new witness assignments (if not in the setup mode).
    ///     (usize, Option<Vec<F>>)
    pub fn transform_lc_map(
        &mut self,
        transformer: &mut dyn FnMut(
            &ConstraintSystem<F>,
            usize,
            &mut LinearCombination<F>,
        ) -> (usize, Option<Vec<F>>),
    ) {
        // `transformed_lc_map` stores the transformed linear combinations.
        let mut transformed_lc_map = BTreeMap::<_, LinearCombination<F>>::new();
        let mut num_times_used = self.lc_num_times_used(false);

        // This loop goes through all the LCs in the map, starting from
        // the early ones. The transformer function is applied to the
        // inlined LC, where new witness variables can be created.
        for (&index, lc) in &self.lc_map {
            let mut transformed_lc = LinearCombination::new();

            // Inline the LC, unwrapping symbolic LCs that may constitute it,
            // and updating them according to transformations in prior iterations.
            for &(coeff, var) in lc.iter() {
                if var.is_lc() {
                    let lc_index = var.get_lc_index().expect("should be lc");

                    // If `var` is a `SymbolicLc`, fetch the corresponding
                    // inlined LC, and substitute it in.
                    //
                    // We have the guarantee that `lc_index` must exist in
                    // `new_lc_map` since a LC can only depend on other
                    // LCs with lower indices, which we have transformed.
                    //
                    let lc = transformed_lc_map
                        .get(&lc_index)
                        .expect("should be inlined");
                    transformed_lc.extend((lc * coeff).0.into_iter());

                    // Delete linear combinations that are no longer used.
                    //
                    // Deletion is safe for both outlining and inlining:
                    // * Inlining: the LC is substituted directly into all use sites, and so once it
                    //   is fully inlined, it is redundant.
                    //
                    // * Outlining: the LC is associated with a new variable `w`, and a new
                    //   constraint of the form `lc_data * 1 = w`, where `lc_data` is the actual
                    //   data in the linear combination. Furthermore, we replace its entry in
                    //   `new_lc_map` with `(1, w)`. Once `w` is fully inlined, then we can delete
                    //   the entry from `new_lc_map`
                    //
                    num_times_used[lc_index.0] -= 1;
                    if num_times_used[lc_index.0] == 0 {
                        // This lc is not used any more, so remove it.
                        transformed_lc_map.remove(&lc_index);
                    }
                } else {
                    // Otherwise, it's a concrete variable and so we
                    // substitute it in directly.
                    transformed_lc.push((coeff, var));
                }
            }
            transformed_lc.compactify();

            // Call the transformer function.
            let (num_new_witness_variables, new_witness_assignments) =
                transformer(&self, num_times_used[index.0], &mut transformed_lc);

            // Insert the transformed LC.
            transformed_lc_map.insert(index, transformed_lc);

            // Update the witness counter.
            self.num_witness_variables += num_new_witness_variables;

            // Supply additional witness assignments if not in the
            // setup mode and if new witness variables are created.
            if !self.is_in_setup_mode() && num_new_witness_variables > 0 {
                assert!(new_witness_assignments.is_some());
                if let Some(new_witness_assignments) = new_witness_assignments {
                    assert_eq!(new_witness_assignments.len(), num_new_witness_variables);
                    self.witness_assignment
                        .extend_from_slice(&new_witness_assignments);
                }
            }
        }
        // Replace the LC map.
        self.lc_map = transformed_lc_map;
    }

    /// Naively inlines symbolic linear combinations into the linear
    /// combinations that use them.
    ///
    /// Useful for standard pairing-based SNARKs where addition gates are cheap.
    /// For example, in the SNARKs such as [\[Groth16\]](https://eprint.iacr.org/2016/260) and
    /// [\[Groth-Maller17\]](https://eprint.iacr.org/2017/540), addition gates
    /// do not contribute to the size of the multi-scalar multiplication, which
    /// is the dominating cost.
    pub fn inline_all_lcs(&mut self) {
        // Only inline when a matrix representing R1CS is needed.
        if !self.should_construct_matrices() {
            return;
        }

        // A dummy closure is used, which means that
        // - it does not modify the inlined LC.
        // - it does not add new witness variables.
        self.transform_lc_map(&mut |_, _, _| (0, None));
    }

    /// If a `SymbolicLc` is used in more than one location and has sufficient
    /// length, this method makes a new variable for that `SymbolicLc`, adds
    /// a constraint ensuring the equality of the variable and the linear
    /// combination, and then uses that variable in every location the
    /// `SymbolicLc` is used.
    ///
    /// Useful for SNARKs like [\[Marlin\]](https://eprint.iacr.org/2019/1047) or
    /// [\[Fractal\]](https://eprint.iacr.org/2019/1076), where addition gates
    /// are not cheap.
    async fn outline_lcs(&mut self) {
        // Only inline when a matrix representing R1CS is needed.
        if !self.should_construct_matrices() {
            return;
        }

        // Store information about new witness variables created
        // for outlining. New constraints will be added after the
        // transformation of the LC map.
        let mut new_witness_linear_combinations = Vec::new();
        let mut new_witness_indices = Vec::new();

        // new code here

        // `transformed_lc_map` stores the transformed linear combinations.
        let mut transformed_lc_map = BTreeMap::<_, LinearCombination<F>>::new();
        let mut num_times_used = self.lc_num_times_used(false);

        // This loop goes through all the LCs in the map, starting from
        // the early ones. The transformer function is applied to the
        // inlined LC, where new witness variables can be created.
        for (&index, lc) in &self.lc_map {
            let mut transformed_lc = LinearCombination::new();

            // Inline the LC, unwrapping symbolic LCs that may constitute it,
            // and updating them according to transformations in prior iterations.
            for &(coeff, var) in lc.iter() {
                if var.is_lc() {
                    let lc_index = var.get_lc_index().expect("should be lc");

                    // If `var` is a `SymbolicLc`, fetch the corresponding
                    // inlined LC, and substitute it in.
                    //
                    // We have the guarantee that `lc_index` must exist in
                    // `new_lc_map` since a LC can only depend on other
                    // LCs with lower indices, which we have transformed.
                    //
                    let lc = transformed_lc_map
                        .get(&lc_index)
                        .expect("should be inlined");
                    transformed_lc.extend((lc * coeff).0.into_iter());

                    // Delete linear combinations that are no longer used.
                    //
                    // Deletion is safe for both outlining and inlining:
                    // * Inlining: the LC is substituted directly into all use sites, and so once it
                    //   is fully inlined, it is redundant.
                    //
                    // * Outlining: the LC is associated with a new variable `w`, and a new
                    //   constraint of the form `lc_data * 1 = w`, where `lc_data` is the actual
                    //   data in the linear combination. Furthermore, we replace its entry in
                    //   `new_lc_map` with `(1, w)`. Once `w` is fully inlined, then we can delete
                    //   the entry from `new_lc_map`
                    //
                    num_times_used[lc_index.0] -= 1;
                    if num_times_used[lc_index.0] == 0 {
                        // This lc is not used any more, so remove it.
                        transformed_lc_map.remove(&lc_index);
                    }
                } else {
                    // Otherwise, it's a concrete variable and so we
                    // substitute it in directly.
                    transformed_lc.push((coeff, var));
                }
            }
            transformed_lc.compactify();

            let num_times_used_val = num_times_used[index.0];
            let inlined_lc = &mut transformed_lc;

            // start of callback

            let mut should_dedicate_a_witness_variable = false;
            let mut new_witness_index = None;
            let mut new_witness_assignment = Vec::new();

            // Check if it is worthwhile to dedicate a witness variable.
            let this_used_times = num_times_used_val + 1;
            let this_len = inlined_lc.len();

            // Cost with no outlining = `lc_len * number of usages`
            // Cost with outlining is one constraint for `(lc_len) * 1 = {new variable}` and
            // using that single new variable in each of the prior usages.
            // This has total cost `number_of_usages + lc_len + 2`
            if this_used_times * this_len > this_used_times + 2 + this_len {
                should_dedicate_a_witness_variable = true;
            }

            // If it is worthwhile to dedicate a witness variable,
            if should_dedicate_a_witness_variable {
                // Add a new witness (the value of the linear combination).
                // This part follows the same logic of `new_witness_variable`.
                let witness_index = self.num_witness_variables;
                new_witness_index = Some(witness_index);

                // Compute the witness assignment.
                if !self.is_in_setup_mode() {
                    let mut acc = F::zero();
                    for (coeff, var) in inlined_lc.iter() {
                        acc += *coeff * &self.assigned_value(*var).await.unwrap();
                    }
                    new_witness_assignment.push(acc);
                }

                // Add a new constraint for this new witness.
                new_witness_linear_combinations.push(inlined_lc.clone());
                new_witness_indices.push(witness_index);

                // Replace the linear combination with (1 * this new witness).
                *inlined_lc = LinearCombination::from(Variable::Witness(witness_index));
            }
            // Otherwise, the LC remains unchanged.

            // Return information about new witness variables.
            let (num_new_witness_variables, new_witness_assignments) =
                if new_witness_index.is_some() {
                    (1, Some(new_witness_assignment))
                } else {
                    (0, None)
                };

            // end of callback

            // Insert the transformed LC.
            transformed_lc_map.insert(index, transformed_lc);

            // Update the witness counter.
            self.num_witness_variables += num_new_witness_variables;

            // Supply additional witness assignments if not in the
            // setup mode and if new witness variables are created.
            if !self.is_in_setup_mode() && num_new_witness_variables > 0 {
                assert!(new_witness_assignments.is_some());
                if let Some(new_witness_assignments) = new_witness_assignments {
                    assert_eq!(new_witness_assignments.len(), num_new_witness_variables);
                    self.witness_assignment
                        .extend_from_slice(&new_witness_assignments);
                }
            }
        }
        // Replace the LC map.
        self.lc_map = transformed_lc_map;

        // end of new code

        // Add the constraints for the newly added witness variables.
        for (new_witness_linear_combination, new_witness_variable) in
            new_witness_linear_combinations
                .iter()
                .zip(new_witness_indices.iter())
        {
            // Add a new constraint
            self.enforce_constraint(
                new_witness_linear_combination.clone(),
                LinearCombination::from(Self::one()),
                LinearCombination::from(Variable::Witness(*new_witness_variable)),
            )
            .unwrap();
        }
    }

    /// Finalize the constraint system (either by outlining or inlining,
    /// if an optimization goal is set).
    pub async fn finalize(&mut self) {
        match self.optimization_goal {
            OptimizationGoal::None => self.inline_all_lcs(),
            OptimizationGoal::Constraints => self.inline_all_lcs(),
            OptimizationGoal::Weight => self.outline_lcs().await,
        };
    }

    /// This step must be called after constraint generation has completed, and
    /// after all symbolic LCs have been inlined into the places that they
    /// are used.
    pub fn to_matrices(&self) -> Option<ConstraintMatrices<F>> {
        if let SynthesisMode::Prove {
            construct_matrices: false,
        } = self.mode
        {
            None
        } else {
            let a: Vec<_> = self
                .a_constraints
                .iter()
                .map(|index| self.make_row(self.lc_map.get(index).unwrap()))
                .collect();
            let b: Vec<_> = self
                .b_constraints
                .iter()
                .map(|index| self.make_row(self.lc_map.get(index).unwrap()))
                .collect();
            let c: Vec<_> = self
                .c_constraints
                .iter()
                .map(|index| self.make_row(self.lc_map.get(index).unwrap()))
                .collect();

            let a_num_non_zero: usize = a.iter().map(|lc| lc.len()).sum();
            let b_num_non_zero: usize = b.iter().map(|lc| lc.len()).sum();
            let c_num_non_zero: usize = c.iter().map(|lc| lc.len()).sum();
            let matrices = ConstraintMatrices {
                num_instance_variables: self.num_instance_variables,
                num_witness_variables: self.num_witness_variables,
                num_constraints: self.num_constraints,

                a_num_non_zero,
                b_num_non_zero,
                c_num_non_zero,

                a,
                b,
                c,
            };
            Some(matrices)
        }
    }

    #[async_recursion(?Send)]
    async fn eval_lc(&self, lc: LcIndex) -> Option<F> {
        let lc = self.lc_map.get(&lc)?;
        let mut acc = F::zero();
        for (coeff, var) in lc.iter() {
            acc += *coeff * self.assigned_value(*var).await?;
        }
        Some(acc)
    }

    /// If `self` is satisfied, outputs `Ok(true)`.
    /// If `self` is unsatisfied, outputs `Ok(false)`.
    /// If `self.is_in_setup_mode()`, outputs `Err(())`.
    pub async fn is_satisfied(&self) -> crate::r1cs::Result<bool> {
        self.which_is_unsatisfied().await.map(|s| s.is_none())
    }

    /// If `self` is satisfied, outputs `Ok(None)`.
    /// If `self` is unsatisfied, outputs `Some(i)`, where `i` is the index of
    /// the first unsatisfied constraint. If `self.is_in_setup_mode()`, outputs
    /// `Err(())`.
    pub async fn which_is_unsatisfied(&self) -> crate::r1cs::Result<Option<String>> {
        if self.is_in_setup_mode() {
            Err(SynthesisError::AssignmentMissing)
        } else {
            for i in 0..self.num_constraints {
                let a = self
                    .eval_lc(self.a_constraints[i])
                    .await
                    .ok_or(SynthesisError::AssignmentMissing)?;
                let b = self
                    .eval_lc(self.b_constraints[i])
                    .await
                    .ok_or(SynthesisError::AssignmentMissing)?;
                let c = self
                    .eval_lc(self.c_constraints[i])
                    .await
                    .ok_or(SynthesisError::AssignmentMissing)?;
                if a * b != c {
                    let trace;
                    #[cfg(feature = "std")]
                    {
                        trace = self.constraint_traces[i].as_ref().map_or_else(
                            || {
                                eprintln!("Constraint trace requires enabling `ConstraintLayer`");
                                format!("{}", i)
                            },
                            |t| format!("{}", t),
                        );
                    }
                    #[cfg(not(feature = "std"))]
                    {
                        trace = format!("{}", i);
                    }
                    return Ok(Some(trace));
                }
            }
            Ok(None)
        }
    }

    /// Obtain the assignment corresponding to the `Variable` `v`.
    pub async fn assigned_value(&self, v: Variable) -> Option<F> {
        match v {
            Variable::One => Some(F::one()),
            Variable::Zero => Some(F::zero()),
            Variable::Witness(idx) => self.witness_assignment.get(idx).copied(),
            Variable::Instance(idx) => self.instance_assignment.get(idx).copied(),
            Variable::SymbolicLc(idx) => {
                let mut lc_assignment_cache = self.lc_assignment_cache.lock().await;
                let value = lc_assignment_cache.get(&idx).copied();
                if value.is_some() {
                    value
                } else {
                    let value = self.eval_lc(idx).await?;
                    lc_assignment_cache.insert(idx, value);
                    Some(value)
                }
            },
        }
    }
}
/// The A, B and C matrices of a Rank-One `ConstraintSystem`.
/// Also contains metadata on the structure of the constraint system
/// and the matrices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintMatrices<F: Field> {
    /// The number of variables that are "public instances" to the constraint
    /// system.
    pub num_instance_variables: usize,
    /// The number of variables that are "private witnesses" to the constraint
    /// system.
    pub num_witness_variables: usize,
    /// The number of constraints in the constraint system.
    pub num_constraints: usize,
    /// The number of non_zero entries in the A matrix.
    pub a_num_non_zero: usize,
    /// The number of non_zero entries in the B matrix.
    pub b_num_non_zero: usize,
    /// The number of non_zero entries in the C matrix.
    pub c_num_non_zero: usize,

    /// The A constraint matrix. This is empty when
    /// `self.mode == SynthesisMode::Prove { construct_matrices = false }`.
    pub a: Matrix<F>,
    /// The B constraint matrix. This is empty when
    /// `self.mode == SynthesisMode::Prove { construct_matrices = false }`.
    pub b: Matrix<F>,
    /// The C constraint matrix. This is empty when
    /// `self.mode == SynthesisMode::Prove { construct_matrices = false }`.
    pub c: Matrix<F>,
}

/// A shared reference to a constraint system that can be stored in high level
/// variables.
#[derive(Debug, Clone)]
pub enum ConstraintSystemRef<F: Field> {
    /// Represents the case where we *don't* need to allocate variables or
    /// enforce constraints. Encountered when operating over constant
    /// values.
    None,
    /// Represents the case where we *do* allocate variables or enforce
    /// constraints.
    CS(Arc<Mutex<ConstraintSystem<F>>>),
}

impl<F: Field> PartialEq for ConstraintSystemRef<F> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::None, Self::None) => true,
            (..) => false,
        }
    }
}

impl<F: Field> Eq for ConstraintSystemRef<F> {}

/// A namespaced `ConstraintSystemRef`.
#[derive(Debug, Clone)]
pub struct Namespace<F: Field> {
    inner: ConstraintSystemRef<F>,
    id: Option<tracing::Id>,
}

impl<F: Field> From<ConstraintSystemRef<F>> for Namespace<F> {
    fn from(other: ConstraintSystemRef<F>) -> Self {
        Self {
            inner: other,
            id: None,
        }
    }
}

impl<F: Field> Namespace<F> {
    /// Construct a new `Namespace`.
    pub fn new(inner: ConstraintSystemRef<F>, id: Option<tracing::Id>) -> Self {
        Self { inner, id }
    }

    /// Obtain the inner `ConstraintSystemRef<F>`.
    pub fn cs(&self) -> ConstraintSystemRef<F> {
        self.inner.clone()
    }

    /// Manually leave the namespace.
    pub fn leave_namespace(self) {
        drop(self)
    }
}

impl<F: Field> Drop for Namespace<F> {
    fn drop(&mut self) {
        if let Some(id) = self.id.as_ref() {
            tracing::dispatcher::get_default(|dispatch| dispatch.exit(id))
        }
        let _ = self.inner;
    }
}

impl<F: Field> ConstraintSystemRef<F> {
    /// Returns `self` if `!self.is_none()`, otherwise returns `other`.
    pub fn or(self, other: Self) -> Self {
        match self {
            ConstraintSystemRef::None => other,
            _ => self,
        }
    }

    /// Returns `true` is `self == ConstraintSystemRef::None`.
    pub fn is_none(&self) -> bool {
        matches!(self, ConstraintSystemRef::None)
    }

    /// Construct a `ConstraintSystemRef` from a `ConstraintSystem`.
    #[inline]
    pub fn new(inner: ConstraintSystem<F>) -> Self {
        Self::CS(Arc::new(Mutex::new(inner)))
    }

    fn inner(&self) -> Option<&Arc<Mutex<ConstraintSystem<F>>>> {
        match self {
            Self::CS(a) => Some(a),
            Self::None => None,
        }
    }

    /// Consumes self to return the inner `ConstraintSystem<F>`. Returns
    /// `None` if `Self::CS` is `None` or if any other references to
    /// `Self::CS` exist.  
    pub fn into_inner(self) -> Option<ConstraintSystem<F>> {
        match self {
            Self::CS(a) => Arc::try_unwrap(a).ok().map(|s| s.into_inner()),
            Self::None => None,
        }
    }

    /// Obtain an immutable reference to the underlying `ConstraintSystem`.
    ///
    /// # Panics
    /// This method panics if `self` is already mutably borrowed.
    #[inline]
    pub async fn borrow(&self) -> Option<MutexGuard<'_, ConstraintSystem<F>>> {
        if let Some(cs) = self.inner() {
            Some(cs.lock().await)
        } else {
            None
        }
    }

    /// Obtain a mutable reference to the underlying `ConstraintSystem`.
    ///
    /// # Panics
    /// This method panics if `self` is already mutably borrowed.
    #[inline]
    pub async fn borrow_mut(&self) -> Option<MutexGuard<'_, ConstraintSystem<F>>> {
        if let Some(cs) = self.inner() {
            Some(cs.lock().await)
        } else {
            None
        }
    }

    /// Check whether `self.mode == SynthesisMode::Setup`.
    #[inline]
    pub async fn is_in_setup_mode(&self) -> bool {
        if let Some(cs) = self.inner() {
            cs.lock().await.is_in_setup_mode()
        } else {
            false
        }
    }

    /// Returns the number of constraints.
    #[inline]
    pub async fn num_constraints(&self) -> usize {
        if let Some(cs) = self.inner() {
            cs.lock().await.num_constraints
        } else {
            0
        }
    }

    /// Returns the number of instance variables.
    #[inline]
    pub async fn num_instance_variables(&self) -> usize {
        if let Some(cs) = self.inner() {
            cs.lock().await.num_instance_variables
        } else {
            0
        }
    }

    /// Returns the number of witness variables.
    #[inline]
    pub async fn num_witness_variables(&self) -> usize {
        if let Some(cs) = self.inner() {
            cs.lock().await.num_witness_variables
        } else {
            0
        }
    }

    /// Check whether this constraint system aims to optimize weight,
    /// number of constraints, or neither.
    #[inline]
    pub async fn optimization_goal(&self) -> OptimizationGoal {
        if let Some(cs) = self.inner() {
            cs.lock().await.optimization_goal()
        } else {
            OptimizationGoal::Constraints
        }
    }

    /// Specify whether this constraint system should aim to optimize weight,
    /// number of constraints, or neither.
    #[inline]
    pub async fn set_optimization_goal(&self, goal: OptimizationGoal) {
        if let Some(cs) = self.inner() {
            cs.lock().await.set_optimization_goal(goal);
        }
    }

    /// Check whether or not `self` will construct matrices.
    #[inline]
    pub async fn should_construct_matrices(&self) -> bool {
        if let Some(cs) = self.inner() {
            cs.lock().await.should_construct_matrices()
        } else {
            false
        }
    }

    /// Obtain a variable representing a new public instance input.
    #[inline]
    pub async fn new_input_variable<Func>(&self, f: Func) -> crate::r1cs::Result<Variable>
    where
        Func: FnOnce() -> crate::r1cs::Result<F> + Send,
    {
        let cs = self.inner().ok_or(SynthesisError::MissingCS)?;

        if !self.is_in_setup_mode().await {
            let value = f();
            cs.lock().await.new_input_variable(|| value)
        } else {
            cs.lock().await.new_input_variable(f)
        }
    }

    /// Obtain a variable representing a new private witness input.
    #[inline]
    pub async fn new_witness_variable<Func>(&self, f: Func) -> crate::r1cs::Result<Variable>
    where
        Func: FnOnce() -> crate::r1cs::Result<F> + Send,
    {
        let cs = self.inner().ok_or(SynthesisError::MissingCS)?;

        if !self.is_in_setup_mode().await {
            let value = f();
            cs.lock().await.new_witness_variable(|| value)
        } else {
            cs.lock().await.new_witness_variable(f)
        }
    }

    /// Obtain a variable representing a linear combination.
    #[inline]
    pub async fn new_lc(&self, lc: LinearCombination<F>) -> crate::r1cs::Result<Variable> {
        let cs = self.inner().ok_or(SynthesisError::MissingCS)?;
        cs.lock().await.new_lc(lc)
    }

    /// Enforce a R1CS constraint with the name `name`.
    #[inline]
    pub async fn enforce_constraint(
        &self,
        a: LinearCombination<F>,
        b: LinearCombination<F>,
        c: LinearCombination<F>,
    ) -> crate::r1cs::Result<()> {
        let cs = self.inner().ok_or(SynthesisError::MissingCS)?;
        cs.lock().await.enforce_constraint(a, b, c)
    }

    /// Naively inlines symbolic linear combinations into the linear
    /// combinations that use them.
    ///
    /// Useful for standard pairing-based SNARKs where addition gates are cheap.
    /// For example, in the SNARKs such as [\[Groth16\]](https://eprint.iacr.org/2016/260) and
    /// [\[Groth-Maller17\]](https://eprint.iacr.org/2017/540), addition gates
    /// do not contribute to the size of the multi-scalar multiplication, which
    /// is the dominating cost.
    pub async fn inline_all_lcs(&self) {
        if let Some(cs) = self.inner() {
            cs.lock().await.inline_all_lcs()
        }
    }

    /// Finalize the constraint system (either by outlining or inlining,
    /// if an optimization goal is set).
    pub async fn finalize(&self) {
        if let Some(cs) = self.inner() {
            cs.lock().await.finalize().await
        }
    }

    /// This step must be called after constraint generation has completed, and
    /// after all symbolic LCs have been inlined into the places that they
    /// are used.
    #[inline]
    pub async fn to_matrices(&self) -> Option<ConstraintMatrices<F>> {
        if let Some(cs) = self.inner() {
            Some(cs.lock().await.to_matrices()?)
        } else {
            None
        }
    }

    /// If `self` is satisfied, outputs `Ok(true)`.
    /// If `self` is unsatisfied, outputs `Ok(false)`.
    /// If `self.is_in_setup_mode()` or if `self == None`, outputs `Err(())`.
    pub async fn is_satisfied(&self) -> crate::r1cs::Result<bool> {
        let cs = self.inner().ok_or(SynthesisError::AssignmentMissing)?;
        cs.lock().await.is_satisfied().await
    }

    /// If `self` is satisfied, outputs `Ok(None)`.
    /// If `self` is unsatisfied, outputs `Some(i)`, where `i` is the index of
    /// the first unsatisfied constraint.
    /// If `self.is_in_setup_mode()` or `self == None`, outputs `Err(())`.
    pub async fn which_is_unsatisfied(&self) -> crate::r1cs::Result<Option<String>> {
        let cs = self.inner().ok_or(SynthesisError::AssignmentMissing)?;
        cs.lock().await.which_is_unsatisfied().await
    }

    /// Obtain the assignment corresponding to the `Variable` `v`.
    pub async fn assigned_value(&self, v: Variable) -> Option<F> {
        let cs = self.inner()?;
        cs.lock().await.assigned_value(v).await
    }

    /// Get trace information about all constraints in the system
    pub async fn constraint_names(&self) -> Option<Vec<String>> {
        #[cfg(feature = "std")]
        {
            let cs = self.inner()?;
            let guard = cs.lock().await;
            guard
                .constraint_traces
                .iter()
                .map(|trace| {
                    let mut constraint_path = String::new();
                    let mut prev_module_path = "";
                    let mut prefixes = ark_std::collections::BTreeSet::new();
                    for step in trace.as_ref()?.path() {
                        let module_path = if prev_module_path == step.module_path {
                            prefixes.insert(step.module_path.to_string());
                            String::new()
                        } else {
                            let mut parts = step
                                .module_path
                                .split("::")
                                .filter(|&part| part != "r1cs_std" && part != "constraints");
                            let mut path_so_far = String::new();
                            for part in parts.by_ref() {
                                if path_so_far.is_empty() {
                                    path_so_far += part;
                                } else {
                                    path_so_far += &["::", part].join("");
                                }
                                if prefixes.contains(&path_so_far) {
                                    continue;
                                } else {
                                    prefixes.insert(path_so_far.clone());
                                    break;
                                }
                            }
                            parts.collect::<Vec<_>>().join("::") + "::"
                        };
                        prev_module_path = step.module_path;
                        constraint_path += &["/", &module_path, step.name].join("");
                    }
                    Some(constraint_path)
                })
                .collect::<Option<Vec<_>>>()
        }
        #[cfg(not(feature = "std"))]
        {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::r1cs::*;

    use ark_ff::One;
    use ark_test_curves::bls12_381::Fr;

    #[tokio::test]
    async fn matrix_generation() -> crate::r1cs::Result<()> {
        let cs = ConstraintSystem::<Fr>::new_ref();
        let two = Fr::one() + Fr::one();
        let a = cs.new_input_variable(|| Ok(Fr::one())).await?;
        let b = cs.new_witness_variable(|| Ok(Fr::one())).await?;
        let c = cs.new_witness_variable(|| Ok(two)).await?;
        cs.enforce_constraint(lc!() + a, lc!() + (two, b), lc!() + c)
            .await?;
        let d = cs.new_lc(lc!() + a + b).await?;
        cs.enforce_constraint(lc!() + a, lc!() + d, lc!() + d)
            .await?;
        let e = cs.new_lc(lc!() + d + d).await?;
        cs.enforce_constraint(lc!() + Variable::One, lc!() + e, lc!() + e)
            .await?;
        cs.inline_all_lcs().await;
        let matrices = cs.to_matrices().await.unwrap();
        assert_eq!(matrices.a[0], vec![(Fr::one(), 1)]);
        assert_eq!(matrices.b[0], vec![(two, 2)]);
        assert_eq!(matrices.c[0], vec![(Fr::one(), 3)]);

        assert_eq!(matrices.a[1], vec![(Fr::one(), 1)]);
        assert_eq!(matrices.b[1], vec![(Fr::one(), 1), (Fr::one(), 2)]);
        assert_eq!(matrices.c[1], vec![(Fr::one(), 1), (Fr::one(), 2)]);

        assert_eq!(matrices.a[2], vec![(Fr::one(), 0)]);
        assert_eq!(matrices.b[2], vec![(two, 1), (two, 2)]);
        assert_eq!(matrices.c[2], vec![(two, 1), (two, 2)]);
        Ok(())
    }

    #[tokio::test]
    async fn matrix_generation_outlined() -> crate::r1cs::Result<()> {
        let cs = ConstraintSystem::<Fr>::new_ref();
        cs.set_optimization_goal(OptimizationGoal::Weight).await;
        let two = Fr::one() + Fr::one();
        let a = cs.new_input_variable(|| Ok(Fr::one())).await?;
        let b = cs.new_witness_variable(|| Ok(Fr::one())).await?;
        let c = cs.new_witness_variable(|| Ok(two)).await?;
        cs.enforce_constraint(lc!() + a, lc!() + (two, b), lc!() + c)
            .await?;

        let d = cs.new_lc(lc!() + a + b).await?;
        cs.enforce_constraint(lc!() + a, lc!() + d, lc!() + d)
            .await?;

        let e = cs.new_lc(lc!() + d + d).await?;
        cs.enforce_constraint(lc!() + Variable::One, lc!() + e, lc!() + e)
            .await?;

        cs.finalize().await;
        assert!(cs.is_satisfied().await.unwrap());
        let matrices = cs.to_matrices().await.unwrap();
        assert_eq!(matrices.a[0], vec![(Fr::one(), 1)]);
        assert_eq!(matrices.b[0], vec![(two, 2)]);
        assert_eq!(matrices.c[0], vec![(Fr::one(), 3)]);

        assert_eq!(matrices.a[1], vec![(Fr::one(), 1)]);
        // Notice here how the variable allocated for d is outlined
        // compared to the example in previous test case.
        // We are optimising for weight: there are less non-zero elements.
        assert_eq!(matrices.b[1], vec![(Fr::one(), 4)]);
        assert_eq!(matrices.c[1], vec![(Fr::one(), 4)]);

        assert_eq!(matrices.a[2], vec![(Fr::one(), 0)]);
        assert_eq!(matrices.b[2], vec![(two, 4)]);
        assert_eq!(matrices.c[2], vec![(two, 4)]);
        Ok(())
    }

    /// Example meant to follow as closely as possible the excellent R1CS
    /// write-up by [Vitalik Buterin](https://vitalik.ca/general/2016/12/10/qap.html)
    /// and demonstrate how to construct such matrices in arkworks.
    #[tokio::test]
    async fn matrix_generation_example() -> crate::r1cs::Result<()> {
        let cs = ConstraintSystem::<Fr>::new_ref();
        // helper definitions
        let three = Fr::from(3u8);
        let five = Fr::from(5u8);
        let nine = Fr::from(9u8);
        // There will be six variables in the system, in the order governed by adding
        // them to the constraint system (Note that the CS is initialised with
        // `Variable::One` in the first position implicitly).
        // Note also that the all public variables will always be placed before all witnesses
        //
        // Variable::One
        // Variable::Instance(35)
        // Variable::Witness(3) ( == x )
        // Variable::Witness(9) ( == sym_1 )
        // Variable::Witness(27) ( == y )
        // Variable::Witness(30) ( == sym_2 )

        // let one = Variable::One; // public input, implicitly defined
        let out = cs
            .new_input_variable(|| Ok(nine * three + three + five))
            .await?; // public input
        let x = cs.new_witness_variable(|| Ok(three)).await?; // explicit witness
        let sym_1 = cs.new_witness_variable(|| Ok(nine)).await?; // intermediate witness variable
        let y = cs.new_witness_variable(|| Ok(nine * three)).await?; // intermediate witness variable
        let sym_2 = cs.new_witness_variable(|| Ok(nine * three + three)).await?; // intermediate witness variable

        cs.enforce_constraint(lc!() + x, lc!() + x, lc!() + sym_1)
            .await?;
        cs.enforce_constraint(lc!() + sym_1, lc!() + x, lc!() + y)
            .await?;
        cs.enforce_constraint(lc!() + y + x, lc!() + Variable::One, lc!() + sym_2)
            .await?;
        cs.enforce_constraint(
            lc!() + sym_2 + (five, Variable::One),
            lc!() + Variable::One,
            lc!() + out,
        )
        .await?;

        cs.finalize().await;
        assert!(cs.is_satisfied().await.unwrap());
        let matrices = cs.to_matrices().await.unwrap();
        // There are four gates(constraints), each generating a row.
        // Resulting matrices:
        // (Note how 2nd & 3rd columns are swapped compared to the online example.
        // This results from an implementation detail of placing all Variable::Instances(_) first.
        //
        // A
        // [0, 0, 1, 0, 0, 0]
        // [0, 0, 0, 1, 0, 0]
        // [0, 0, 1, 0, 1, 0]
        // [5, 0, 0, 0, 0, 1]
        // B
        // [0, 0, 1, 0, 0, 0]
        // [0, 0, 1, 0, 0, 0]
        // [1, 0, 0, 0, 0, 0]
        // [1, 0, 0, 0, 0, 0]
        // C
        // [0, 0, 0, 1, 0, 0]
        // [0, 0, 0, 0, 1, 0]
        // [0, 0, 0, 0, 0, 1]
        // [0, 1, 0, 0, 0, 0]
        assert_eq!(matrices.a[0], vec![(Fr::one(), 2)]);
        assert_eq!(matrices.b[0], vec![(Fr::one(), 2)]);
        assert_eq!(matrices.c[0], vec![(Fr::one(), 3)]);

        assert_eq!(matrices.a[1], vec![(Fr::one(), 3)]);
        assert_eq!(matrices.b[1], vec![(Fr::one(), 2)]);
        assert_eq!(matrices.c[1], vec![(Fr::one(), 4)]);

        assert_eq!(matrices.a[2], vec![(Fr::one(), 2), (Fr::one(), 4)]);
        assert_eq!(matrices.b[2], vec![(Fr::one(), 0)]);
        assert_eq!(matrices.c[2], vec![(Fr::one(), 5)]);

        assert_eq!(matrices.a[3], vec![(five, 0), (Fr::one(), 5)]);
        assert_eq!(matrices.b[3], vec![(Fr::one(), 0)]);
        assert_eq!(matrices.c[3], vec![(Fr::one(), 1)]);
        Ok(())
    }
}
