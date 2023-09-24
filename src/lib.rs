#![feature(map_try_insert)]

pub mod error;
pub mod ast_verbose_print;
pub mod jit_constraint;
pub mod ast_metadata;
pub mod util;

#[cfg(test)]
mod tests {
    use z3::ast::{BV, Ast, Dynamic};

    use crate::ast_verbose_print::VerbosePrint;

    #[test]
    fn it_works() {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(60000);
        let ctx = z3::Context::new(&mut cfg);
        let _one = BV::from_u64(&ctx, 1, 8);
        let var0 = BV::new_const(&ctx, "var0", 8);
        let var1 = BV::new_const(&ctx, "var1", 8);

        let cst = (&var0 + &var1)._eq(&_one);
        let _dyn : Dynamic = cst.clone().into();
        _dyn.print_verbose();

        // expr1 = <some really complicated stuff, expensive to compute
        // (expr1 + expr1)
        // (expr1 + expr1) * expr1
        // cst: var0 + var1 == 1
        // to_compute: input_byte[0] + input_byte[1] == 1
        // (= (+ var0 var1) 1)
        // let y = "
        // char model[0x1000];

        // int tmp0 = model[0] + model[1];
        // bool tmp1 = tmp0 == 1;
        // return tmp1;
        // ";

    }
}
