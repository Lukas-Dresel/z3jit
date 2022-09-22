macro_rules! function_name {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);
        &name[..name.len() - 3]
    }}
}

macro_rules! timeit {
    ($e:expr) => {
        {
            let clk = unsafe { core::arch::x86_64::_rdtsc() };
            let res = $e;
            let endclk = unsafe { core::arch::x86_64::_rdtsc() };
            (endclk - clk, res)
        }
    }
}

pub(crate) use function_name;
pub(crate) use timeit;

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
