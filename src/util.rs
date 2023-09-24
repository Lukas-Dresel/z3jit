use z3::SatResult;
use z3::ast::Bool;
use std::time::Instant;

pub trait TimedSolver<'ctx> {
    fn timed_check_assumptions(&self, assumptions: &[&Bool<'ctx>]) -> SatResult;
}

impl<'ctx> TimedSolver<'ctx> for z3::Solver<'ctx> {
    fn timed_check_assumptions(&self, assumptions: &[&Bool<'ctx>]) -> SatResult {
        println!("Calling Solver.check([{} assumptions])", assumptions.len());
        let start = Instant::now();
        let res = self.check_assumptions(assumptions);
        let duration = start.elapsed();
        println!("Solver.check([{} assumptions]) = {:?} took {:?}", assumptions.len(), res, duration);
        res
    }
}

impl<'ctx> TimedSolver<'ctx> for z3::Optimize<'ctx> {
    fn timed_check_assumptions(&self, assumptions: &[&Bool<'ctx>]) -> SatResult {
        println!("Calling Optimize.check([{} assumptions])", assumptions.len());
        let start = Instant::now();
        let res = self.check(assumptions);
        let duration = start.elapsed();
        println!("Optimize.check([{} assumptions]) = {:?} took {:?}", assumptions.len(), res, duration);
        res
    }
}
