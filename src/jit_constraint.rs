use std::fmt::{Display, Debug};
use std::collections::HashMap;

use inkwell::types::{PointerType, StringRadix};
use itertools::Itertools;

use z3::SortKind;
use z3::ast::Bool;
use z3::{ast::{BV, Ast, Dynamic}, SatResult};

use inkwell::{OptimizationLevel, AddressSpace};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{BasicValue, AnyValue, IntValue, PointerValue};

use crate::ast_verbose_print::VerbosePrint;

fn topologically_sorted_nodes<'z3ctx>(cst: &Dynamic<'z3ctx>) -> Vec<Dynamic<'z3ctx>> {
    let mut result_vec: Vec<Dynamic> = vec![];
    for child in cst.children() {
        result_vec.append(&mut topologically_sorted_nodes(&child));
    }
    result_vec.push(cst.clone());
    let result = result_vec.into_iter().unique().collect();
    result
}

struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}
type JittedValues<'z3ctx, 'llvmctx> = HashMap<Dynamic<'z3ctx>, Box<dyn AnyValue<'llvmctx>>>;
type JittedConstraintFunc = unsafe extern "C" fn(*const u8, i64) -> bool;

impl<'z3ctx, 'llvmctx> CodeGen<'llvmctx> {

    fn compile_bool_app(
        &self,
        ast: Bool<'z3ctx>,
        jitted_values: JittedValues<'z3ctx, 'llvmctx>,
        input_buffer: PointerValue<'llvmctx>,
        buffer_size: IntValue<'llvmctx>,
    ) -> Result<(IntValue<'llvmctx>, JittedValues<'z3ctx, 'llvmctx>), String> {

        todo!()
    }
    fn compile_bv_numeral(
        &self,
        bv: BV<'z3ctx>,
        jitted_values: JittedValues<'z3ctx, 'llvmctx>,
        input_buffer: PointerValue<'llvmctx>,
        buffer_size: IntValue<'llvmctx>,
    ) -> Result<(IntValue<'llvmctx>, JittedValues<'z3ctx, 'llvmctx>), String> {
        let sort = bv.get_sort();
        let size = sort.bv_size().ok_or("Could not retrieve bitvector size.")?;
        assert!(size <= 128);
        let llvm_val = match size {
            64 | 32 | 16 | 8 => {
                let val = bv.as_u64().expect(&format!("Could not get numeral value of {:?}", bv));
                match size {
                    64 => self.context.i64_type(),
                    32 => self.context.i32_type(),
                    16 => self.context.i16_type(),
                    8 => self.context.i8_type(),
                    _ => unreachable!()
                }.const_int(val, false)
            }
            custom_bitwidth => {
                let val = bv.as_numeral_string().expect(&format!("Could not get numeral string value for {:?}.", bv));
                self.context
                    .custom_width_int_type(
                        custom_bitwidth
                            .try_into()
                            .unwrap()
                    )
                    .const_int_from_string(&val, StringRadix::Decimal).unwrap()
            }
        };
        Ok((llvm_val, jitted_values))
    }
    fn compile_bv_app(
        &self,
        bv: BV<'z3ctx>,
        jitted_values: JittedValues<'z3ctx, 'llvmctx>,
        input_buffer: PointerValue<'llvmctx>,
        buffer_size: IntValue<'llvmctx>,
    ) -> Result<(IntValue<'llvmctx>, JittedValues<'z3ctx, 'llvmctx>), String> {
        todo!()
    }
    fn compile_ast(
        &self,
        ast: Dynamic<'z3ctx>,
        jitted_values: JittedValues<'z3ctx, 'llvmctx>,
        input_buffer: PointerValue<'llvmctx>,
        buffer_size: IntValue<'llvmctx>,
    ) -> Result<(IntValue<'llvmctx>, JittedValues<'z3ctx, 'llvmctx>), String>
    {
        ast.print_verbose();
        match ast.sort_kind() {
            z3::SortKind::Bool => {
                let _bool = ast.as_bool().unwrap();
                match ast.kind() {
                    z3::AstKind::App => self.compile_bool_app(_bool, jitted_values, input_buffer, buffer_size),
                    _ => todo!()
                }
            },
            z3::SortKind::BV => {
                let bv = ast.as_bv().ok_or("Could not get bitvec representation of BV sort AST?")?;
                match ast.kind() {
                    z3::AstKind::Numeral => self.compile_bv_numeral(bv, jitted_values, input_buffer, buffer_size),
                    z3::AstKind::App => self.compile_bv_app(bv, jitted_values, input_buffer, buffer_size),
                    _ => todo!()
                }
            },
            _ => todo!()
        }

    }
    fn jit_compile_constraint(&self, function_name: &str, cst: Bool<'z3ctx>) -> Option<JitFunction<JittedConstraintFunc>> {
        let i64_type = self.context.i64_type();
        let byte_type = self.context.i8_type();
        let buf_type = byte_type.ptr_type(AddressSpace::Const);
        let bool_type = self.context.bool_type();

        let fn_type = bool_type.fn_type(&[buf_type.into(), i64_type.into()], false);
        let function = self.module.add_function(function_name, fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        let buffer = function.get_nth_param(0)?.into_pointer_value();
        let buffer_size = function.get_nth_param(1)?.into_int_value();

        let (result, values) = self.compile_ast(
            cst.into(),
            Default::default(),
            buffer,
            buffer_size
        )
            .unwrap();

        self.builder.build_return(Some(&result));

        unsafe { self.execution_engine.get_function(function_name).ok() }
    }
}

#[cfg(test)]
mod tests {
    use inkwell::{OptimizationLevel};
    use z3::ast::{BV, Ast, Dynamic};

    use super::CodeGen;


    #[test]
    fn test_empty() {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(60000);
        let ctx = z3::Context::new(&mut cfg);
        let _one = BV::from_u64(&ctx, 1, 8);
        let var0 = BV::new_const(&ctx, "var0", 8);
        let var1 = BV::new_const(&ctx, "var1", 8);

        let add = &var0 + &var1;
        let add2 = &add + &var0;
        let cst = add2._eq(&_one);

        let cst = _one._eq(&_one);

        let ctx = inkwell::context::Context::create();
        let module = ctx.create_module("jitted_constraint");
        let builder = ctx.create_builder();
        let eng = module.create_jit_execution_engine(OptimizationLevel::Aggressive).unwrap();

        let code_gen = CodeGen {
            context: &ctx,
            module,
            builder,
            execution_engine: eng,
        };

        let result = code_gen.jit_compile_constraint("cst1", cst).unwrap();
        println!("Successfully jitted: {:?}", result);
        unsafe {
            let input_buf= [0u8; 2];
            let res = result.call(input_buf.as_ptr(), 2);
            println!("Jitted function {:?} returned {:?}", result, res);
        }

    }
}