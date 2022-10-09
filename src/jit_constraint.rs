use std::fmt::{Debug};
use std::collections::HashMap;
use std::path::Path;

use inkwell::types::{PointerType, StringRadix, FunctionType, IntType};
use itertools::Itertools;

use z3::{FuncDecl, DeclParam};
use z3::ast::Bool;
use z3::{ast::{BV, Ast, Dynamic}, SatResult};

use inkwell::{OptimizationLevel, AddressSpace, IntPredicate};
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::values::{IntValue, PointerValue, FunctionValue, BasicValueEnum};

use crate::ast_metadata::AstMetadata;
use crate::util::TimedSolver;

fn topologically_sorted_nodes<'z3ctx>(cst: &Dynamic<'z3ctx>) -> Vec<Dynamic<'z3ctx>> {
    let mut result_vec: Vec<Dynamic> = vec![];
    for child in cst.children() {
        result_vec.append(&mut topologically_sorted_nodes(&child));
    }
    result_vec.push(cst.clone());
    let result = result_vec.into_iter().unique().collect();
    result
}

pub type JitContext = inkwell::context::Context;

type JittedValues<'llvmctx, 'z3ctx> = HashMap<Dynamic<'z3ctx>, IntValue<'llvmctx>>;
type JittedConstraintFunc = unsafe extern "C" fn(*const u8, i64) -> bool;

#[derive(Debug)]
pub struct CodeGen<'llvmctx, 'z3ctx> {
    context: &'llvmctx Context,
    module: Module<'llvmctx>,
    builder: Builder<'llvmctx>,
    execution_engine: ExecutionEngine<'llvmctx>,

    fn_name: String,
    fn_type: FunctionType<'llvmctx>,
    function: FunctionValue<'llvmctx>,

    input_buffer_type: PointerType<'llvmctx>,
    buffer_size_type: IntType<'llvmctx>,
    input_buffer: PointerValue<'llvmctx>,
    buffer_size: IntValue<'llvmctx>,

    jitted_values: JittedValues<'llvmctx, 'z3ctx>,

    csts: Vec<Dynamic<'z3ctx>>,
    ast_meta: AstMetadata<'z3ctx>,
}

impl<'llvmctx, 'z3ctx> CodeGen<'llvmctx, 'z3ctx> {
    pub fn new(context: &'llvmctx Context, asts: Vec<Bool<'z3ctx>>, name: &str) -> CodeGen<'llvmctx, 'z3ctx> {
        let module = context.create_module("module");
        let builder = context.create_builder();
        let eng = module.create_jit_execution_engine(OptimizationLevel::Aggressive).unwrap();

        // Workaround for LTO "JIT has not been linked in." issue, see https://github.com/TheDan64/inkwell/issues/320
        // This prevents rustc from optimizing out the actual JIT implementation
        ExecutionEngine::link_in_mc_jit();
        // ExecutionEngine::link_in_interpreter();

        let i64_type = context.i64_type();
        let byte_type = context.i8_type();
        let buf_type = byte_type.ptr_type(AddressSpace::Generic);
        let bool_type = context.bool_type();

        let fn_type = bool_type.fn_type(&[buf_type.into(), i64_type.into()], false);
        let function = module.add_function(name, fn_type, None);
        let basic_block_entry = context.append_basic_block(function, "entry");
        builder.position_at_end(basic_block_entry);

        let buffer = function.get_nth_param(0).expect("huh? we defined these args").into_pointer_value();
        let buffer_size = function.get_nth_param(1).expect("huh? we defined this arg too").into_int_value();

        let dyn_asts: Vec<Dynamic> = asts.iter().map(|x|x.clone().into()).collect();
        let ast_meta = AstMetadata::from(&dyn_asts[..]);
        let asts_sorted = asts
            .into_iter()
            .sorted_by_cached_key(|x| ast_meta.ast_depths.get(&x.clone().into())).collect::<Vec<_>>();
        let mut code_gen = CodeGen {
            context,
            module,
            builder,
            execution_engine: eng,

            fn_name: String::from(name),
            fn_type,
            function,

            input_buffer_type: buf_type,
            buffer_size_type: i64_type,
            input_buffer: buffer,
            buffer_size,

            jitted_values: HashMap::new(),

            csts: dyn_asts,
            ast_meta,
        };
        let buf_size_min = code_gen.buffer_size_type.const_int(code_gen.ast_meta.max_byte_index.try_into().unwrap(), true);
        let buf_size_predicate = code_gen.builder.build_int_compare(
            IntPredicate::SGE,
            buffer_size,
            buf_size_min,
            "buffer_size_check"
        );
        code_gen.build_early_exit_false(buf_size_predicate, "buffer_too_small");
        for c in asts_sorted
        {
            code_gen.compile_constraint(c).unwrap();
        }
        let retval = code_gen.context.bool_type().const_int(1, false);
        code_gen.builder.build_return(Some(&retval));
        code_gen.execution_engine.get_function_address(name).unwrap(); // check that it worked for now like this

        code_gen
    }

    fn build_early_exit_false(&mut self, pred_to_check: IntValue<'llvmctx>, exit_id: &str) {
        let return_basic_block = self.context.append_basic_block(self.function, &format!("early_exit_{}", exit_id));
        let continuation_basic_block = self.context.append_basic_block(self.function, &format!("continuation_{}", exit_id));
        self.builder.build_conditional_branch(pred_to_check, continuation_basic_block, return_basic_block);
        self.builder.position_at_end(return_basic_block);
        self.builder.build_return(Some(&self.context.bool_type().const_int(0, false)));
        self.builder.position_at_end(continuation_basic_block);
    }
    fn get_function(&self) -> Option<JitFunction<JittedConstraintFunc>> {
        let func = unsafe { self.execution_engine.get_function(&self.fn_name).ok() };
        func
    }

    pub fn evaluate_input(&self, input: &[u8]) -> bool {
        let func = self.get_function().unwrap();
        unsafe {
            func.call(input.as_ptr(), input.len() as i64)
        }
    }

    pub fn dump_bitcode(&self, path: &str) {
        self.module.write_bitcode_to_path(Path::new(path));
    }

    #[inline]
    fn compile_children(&mut self, ast: &dyn Ast<'z3ctx>) -> Result<Vec<IntValue<'llvmctx>>, String> {
        ast
            .children()
            .into_iter()
            .map(|child| self.compile_ast(child))
            .collect::<Result<Vec<_>, _>>()
    }
    fn compile_bool_app(
        &mut self,
        ast: Bool<'z3ctx>,
    ) -> Result<IntValue<'llvmctx>, String> {
        let decl = ast.decl();
        macro_rules! z3_intcmp_to_llvm {
            ($predicate:expr) => {
                {
                    let llvm_children = self.compile_children(&ast)?;
                    assert!(llvm_children.len() == 2);
                    let (lhs, rhs) = (llvm_children[0], llvm_children[1]);
                    let cmp_result = self.builder.build_int_compare($predicate, lhs, rhs, "");
                    cmp_result
                }
            }
        }

        macro_rules! z3_bool_starop_to_llvm {
            ($function:ident) => {
                {
                    let llvm_children = self.compile_children(&ast)?;
                    assert!(llvm_children.len() >= 2, "bit vector addition with less than 2 children: {:?}, {:?}", ast, llvm_children);
                    let init = llvm_children[0];

                    let folded_children = llvm_children[1..]
                        .into_iter()
                        .fold(init, |old, &cur| {
                            self.builder.$function(old, cur, "")
                        });
                    folded_children
                }
            }
        }

        let llvm_result = match decl.kind() {
            z3::DeclKind::NOT => {
                let llvm_children = self.compile_children(&ast)?;
                assert!(llvm_children.len() == 1);
                self.builder.build_not(llvm_children[0], "")
            },

            z3::DeclKind::EQ => z3_intcmp_to_llvm!(IntPredicate::EQ),

            z3::DeclKind::ULEQ => z3_intcmp_to_llvm!(IntPredicate::ULE),
            z3::DeclKind::ULT => z3_intcmp_to_llvm!(IntPredicate::ULT),
            z3::DeclKind::UGEQ => z3_intcmp_to_llvm!(IntPredicate::UGE),
            z3::DeclKind::UGT => z3_intcmp_to_llvm!(IntPredicate::UGT),

            z3::DeclKind::SLEQ => z3_intcmp_to_llvm!(IntPredicate::SLE),
            z3::DeclKind::SLT => z3_intcmp_to_llvm!(IntPredicate::SLT),
            z3::DeclKind::SGEQ => z3_intcmp_to_llvm!(IntPredicate::SGE),
            z3::DeclKind::SGT => z3_intcmp_to_llvm!(IntPredicate::SGT),

            z3::DeclKind::AND => z3_bool_starop_to_llvm!(build_and),
            z3::DeclKind::OR => z3_bool_starop_to_llvm!(build_or),

            z3::DeclKind::TRUE => self.context.bool_type().const_int(1, false),
            z3::DeclKind::FALSE => self.context.bool_type().const_int(0, false),

            _ => todo!("Haven't implemented this yet!! {:?}", decl)
        };
        Ok(llvm_result)
    }
    fn llvm_int_type(&self, bitwidth: usize) -> IntType<'llvmctx> {
        match bitwidth {
            64 | 32 | 16 | 8 => {
                match bitwidth {
                    64 => self.context.i64_type(),
                    32 => self.context.i32_type(),
                    16 => self.context.i16_type(),
                    8 => self.context.i8_type(),
                    _ => unreachable!()
                }
            }
            custom_bitwidth => {
                self.context
                    .custom_width_int_type(
                        custom_bitwidth
                            .try_into()
                            .unwrap()
                    )
            }
        }
    }
    fn compile_bv_numeral(
        &mut self,
        bv: BV<'z3ctx>,
    ) -> Result<IntValue<'llvmctx>, String> {
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
        Ok(llvm_val)
    }
    fn compile_bv_app(
        &mut self,
        bv: BV<'z3ctx>,
    ) -> Result<IntValue<'llvmctx>, String> {
        macro_rules! z3_binop_to_llvm {
            ($function:tt) => {
                {
                    let llvm_children = self.compile_children(&bv)?;
                    assert!(llvm_children.len() == 2, "bit vector addition with non-2 children: {:?}, {:?}", bv, llvm_children);
                    Ok(self.builder.$function(llvm_children[0], llvm_children[1], ""))
                }
            }
        }
        macro_rules! z3_starop_to_llvm {
            ($function:tt) => {
                {
                    let llvm_children = self.compile_children(&bv)?;
                    assert!(llvm_children.len() >= 2, "bit vector addition with less than 2 children: {:?}, {:?}", bv, llvm_children);
                    let init = llvm_children[0];

                    let folded_children = llvm_children[1..]
                        .into_iter()
                        .fold(init, |old, &cur| {
                            self.builder.$function(old, cur, "")
                        });
                    Ok(folded_children)
                }
            }
        }
        let decl = bv.decl();
        let i64type = self.context.i64_type();
        match decl.kind() {
            z3::DeclKind::UNINTERPRETED => {
                assert!(decl.arity() == 0);
                // variable, grab from input_buffer

                let &index = self.ast_meta.variable_to_byte.get(&decl).expect("All variables should have been found in the index!");
                // println!("Accessing index: {:?}[{}](max={:?})", self.input_buffer, index, self.buffer_size);
                let llvm_idx = i64type.const_int(index.try_into().unwrap(), false);
                let typ = self.input_buffer.get_type();
                // println!("input_buffer: {:?}, type: {:?} {:?}", self.input_buffer, typ, typ.print_to_string());
                let element_ptr = unsafe {
                    self.builder.build_gep(self.input_buffer, &[llvm_idx], "gep")
                };
                let result = self.builder.build_load(element_ptr, &format!("var_{}", index));
                if let BasicValueEnum::IntValue(x) = result {
                    Ok(x)
                }
                else {
                    unreachable!(
                        "Loading an input variable resulted in a non-int value! input_buffer={:?}, llvm_idx={:?}, element_ptr={:?}, result={:?}",
                        self.input_buffer,
                        llvm_idx,
                        element_ptr,
                        result)
                }
            },
            z3::DeclKind::EXTRACT => {
                let llvm_children = self.compile_children(&bv)?;
                assert!(llvm_children.len() == 1);
                let child = llvm_children[0];

                let params = decl.params();
                assert!(params.len() == 2);
                match (params[0].clone(), params[1].clone()) {
                    (DeclParam::Int(hi), DeclParam::Int(lo)) => {
                        assert!(hi >= lo);
                        assert!(hi >= 0 && lo >= 0);
                        let result_type = self.llvm_int_type((hi - lo + 1).try_into().unwrap());
                        let shift_amount = child
                            .get_type()
                            .const_int(lo.try_into().unwrap(), false);
                        let shifted = self.builder
                            .build_right_shift(child, shift_amount, false, "");
                        let truncated = self.builder
                            .build_int_truncate(shifted, result_type, "");
                        Ok(truncated)
                    }
                    _ => {
                        unreachable!("Somehow the extract parameters are bogus?? {:?}", params);
                    }
                }
                // self.builder.build_int_truncate(vector, index, name)
            },
            z3::DeclKind::CONCAT => {
                {
                    let llvm_children = self.compile_children(&bv)?;
                    assert!(llvm_children.len() >= 2, "bit vector addition with less than 2 children: {:?}, {:?}", bv, llvm_children);
                    let init = llvm_children[0];

                    let folded_children = llvm_children[1..]
                        .into_iter()
                        .fold(init, |old, &cur| {

                            let rhs_bitsize = cur.get_type().get_bit_width() as usize;
                            let lhs_bitsize = old.get_type().get_bit_width() as usize;
                            let result_type = self.llvm_int_type(lhs_bitsize + rhs_bitsize);

                            let lhs = self.builder.build_int_z_extend(old, result_type, "");
                            let rhs = self.builder.build_int_z_extend(cur, result_type, "");

                            let shift_amount = result_type.const_int(rhs_bitsize.try_into().unwrap(), false);
                            let lhs_shifted = self.builder.build_left_shift(lhs, shift_amount, "");

                            let result = self.builder.build_or(lhs_shifted, rhs, "");
                            result
                        });
                    Ok(folded_children)
                }
            }
            z3::DeclKind::BADD => z3_starop_to_llvm!(build_int_add),
            z3::DeclKind::BSUB => z3_binop_to_llvm!(build_int_sub),
            z3::DeclKind::BMUL => z3_starop_to_llvm!(build_int_mul),
            z3::DeclKind::BAND => z3_starop_to_llvm!(build_and),
            z3::DeclKind::BOR => z3_starop_to_llvm!(build_or),
            z3::DeclKind::BXOR => z3_starop_to_llvm!(build_xor),
            z3::DeclKind::BNOT => {
                let llvm_children = self.compile_children(&bv)?;
                assert!(llvm_children.len() == 1);
                let child = llvm_children[0];
                let result = self.builder.build_not(child, "");
                Ok(result)
            },
            z3::DeclKind::ITE => {
                let llvm_children = self.compile_children(&bv)?;
                assert!(llvm_children.len() == 3, "bit vector ITE with non-3 children: {:?}, {:?}", bv, llvm_children);
                let id = bv.get_ast_id();
                let res = self.builder.build_select(llvm_children[0], llvm_children[1], llvm_children[2], &format!("ite_{}", id));
                match res {
                    BasicValueEnum::IntValue(iv) => Ok(iv),
                    x => todo!("ITE result is not an int value: {:?}", x),
                }
            },
            div @ (z3::DeclKind::BUDIV_I | z3::DeclKind::BSDIV_I | z3::DeclKind::BSREM_I) => {
                let llvm_children = self.compile_children(&bv)?;
                assert!(llvm_children.len() == 2, "bit vector integer division with non-2 children: {:?}, {:?}", bv, llvm_children);
                // println!("BUDIV_I: bv: {:?}, llvm_children: {:?}", bv, llvm_children);
                let id = bv.get_ast_id();

                let result = match div {
                    z3::DeclKind::BUDIV_I => {
                        self.builder.build_int_unsigned_div(llvm_children[0], llvm_children[1], &format!("udiv_{}", id))
                    },
                    z3::DeclKind::BSDIV_I => {
                        self.builder.build_int_signed_div(llvm_children[0], llvm_children[1], &format!("sdiv_{}", id))
                    },
                    z3::DeclKind::BSREM_I => {
                        self.builder.build_int_signed_rem(llvm_children[0], llvm_children[1], &format!("srem_{}", id))
                    }
                    _ => unreachable!(),
                };
                // self.dump_bitcode("bitcode_after_div");
                Ok(result)
            },
            shift @ (z3::DeclKind::BASHR | z3::DeclKind::BLSHR | z3::DeclKind::BSHL) => {
                let llvm_children = self.compile_children(&bv)?;
                assert!(llvm_children.len() == 2, "bit vector arithmetic shift right with non-2 children: {:?}, {:?}", bv, llvm_children);
                let id = bv.get_ast_id();
                let result = match shift {
                    z3::DeclKind::BASHR => {
                        self.builder.build_right_shift(llvm_children[0], llvm_children[1], true, &format!("ashr_{}", id))
                    },
                    z3::DeclKind::BLSHR => {
                        self.builder.build_right_shift(llvm_children[0], llvm_children[1], false, &format!("lshr_{}", id))
                    },
                    z3::DeclKind::BSHL => {
                        self.builder.build_left_shift(llvm_children[0], llvm_children[1], &format!("shl_{}", id))
                    },
                    _ => unreachable!(),
                };
                Ok(result)
            },
            other => todo!("Have not yet implemented BV operation {:?} [{:?}]", decl, other)
        }
    }

    fn compile_ast(
        &mut self,
        ast: Dynamic<'z3ctx>,
    ) -> Result<IntValue<'llvmctx>, String>
    {
        if let Some(&x) = self.jitted_values.get(&ast) {
            return Ok(x)
        }

        let llvm_result = match ast.sort_kind() {
            z3::SortKind::Bool => {
                let _bool = ast.as_bool().unwrap();
                match ast.kind() {
                    z3::AstKind::App => self.compile_bool_app(_bool),
                    _ => todo!()
                }
            },
            z3::SortKind::BV => {
                let bv = ast.as_bv().ok_or("Could not get bitvec representation of BV sort AST?")?;
                match ast.kind() {
                    z3::AstKind::Numeral => self.compile_bv_numeral(bv),
                    z3::AstKind::App => self.compile_bv_app(bv),
                    _ => todo!()
                }
            },
            _ => todo!("Didn't implement operations of SortKind {:?} yet", ast.sort_kind())
        }?;
        // println!("Cache: {:?}, ast: {:?}, result: {:?}", self.jitted_values, ast, llvm_result);
        match self.jitted_values.insert(ast.clone().into(), llvm_result) {
            Some(prev) => Err(format!("Cache already contained item {:?}, prev={:?}", ast, prev)),
            None => Ok(llvm_result)
        }
    }
    fn compile_constraint(
        &mut self,
        cst: Bool<'z3ctx>,
    ) -> Result<(), String>
    {
        let id = cst.get_ast_id();
        // println!("Compiling constraint {:?}[id={:x}]", ast, id);
        let comparison_result = self.compile_ast(cst.into())?;
        self.build_early_exit_false(
            comparison_result,
            &format!("constraint_not_sat_{}", id)
        );

        Ok(())
    }
}

fn get_bytes_model<'z3ctx, 'slice>(ctx: &z3::Context, ast_meta: &AstMetadata<'z3ctx>, csts: &[Bool<'z3ctx>]) -> Option<Vec<u8>> {
    let solver = z3::Solver::new(ctx);
    for cst in csts {
        solver.assert(cst);
    }

    if solver.timed_check_assumptions(&[]) != SatResult::Sat {
        return None
    }
    let model = solver.get_model().unwrap();

    let num_bytes = ast_meta.max_byte_index;

    let mut input_bytes: Vec<u8> = (0..num_bytes).map(|_| 0u8).collect();
    for (decl, byte_idx) in ast_meta.variable_to_byte.iter() {
        if let Some(x) = model.get_const_interpretation(decl) {
            let val = x.as_bv().unwrap().as_u64().unwrap().try_into().unwrap();
            input_bytes[*byte_idx] = val;
        }
        else
        {
            input_bytes[*byte_idx] = rand::random();
        }
    }
    Some(input_bytes)
}

#[cfg(test)]
mod tests {
    use std::{time::{Instant, Duration}, path::Path, process::Command};
    use z3::ast::{BV, Ast, Bool, Dynamic};
    use crate::{util::{function_name, timeit}, ast_metadata::AstMetadata};
    use super::{CodeGen, get_bytes_model};


    fn jit_constraint_test<'ctx>(caller: &str, csts: Vec<Bool<'ctx>>, tests: &[(Vec<u8>, bool)]) {
        // let _dyn: Dynamic = cst.clone().into();
        // _dyn.print_verbose();

        let ctx = inkwell::context::Context::create();
        let (duration, code_gen) = timeit!({
            CodeGen::new(&ctx, csts, "jitted_constraint")
        });
        println!("Successfully jitted! It took {:?}", duration);

        let bitcode_path = format!("./test_data/{}.bc", caller);
        code_gen.dump_bitcode(&bitcode_path);
        Command::new("llvm-dis").arg(&bitcode_path).output().unwrap();
        std::fs::remove_file(Path::new(&bitcode_path)).unwrap();

        // let mut buffer = String::new();
        // std::io::stdin().read_line(&mut buffer).unwrap();

        for (input, output) in tests.into_iter() {
            unsafe {
                let result = code_gen.get_function().unwrap();
                result.call(input.as_ptr(), 2);
                let (durations, results) : (Vec<u64>, Vec<bool>) = (0..10)
                    .fold((vec![], vec![]), |(mut durs, mut ress), _| {
                        let (d, r) = timeit!(result.call(input.as_ptr(), 2));
                        durs.push(d);
                        ress.push(r);
                        (durs, ress)
                    });
                println!("Running the jitted function took {:?} cycles and returned {:?}[expected:{:?}]", durations, results, *output);
                for v in results{
                    assert_eq!(*output, v);
                }
            }
        }
    }

    #[test]
    fn test_simple_var_adds() {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(60000);
        let ctx = z3::Context::new(&mut cfg);
        let _one = BV::from_u64(&ctx, 1, 8);
        let var0 = BV::new_const(&ctx, "var!00", 8);
        let var1 = BV::new_const(&ctx, "var!10", 8);

        let add = &var0 + &var1;
        let add2 = &add + &var0;
        let cst = add2._eq(&_one);
        jit_constraint_test(&function_name!(), vec![cst], &[
            (vec![1, 0], false),
            (vec![0, 1], true),
        ]);
    }
    #[test]
    fn test_simple_var_subs() {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(60000);
        let ctx = z3::Context::new(&mut cfg);
        let _one = BV::from_u64(&ctx, 1, 8);
        let var0 = BV::new_const(&ctx, "var!00", 8);
        let var1 = BV::new_const(&ctx, "var!10", 8);

        let add = &var0 - &var1;
        let add2 = &add - &var1;
        let cst = add2._eq(&_one);
        jit_constraint_test(&function_name!(), vec![cst], &[
            (vec![3, 2], false),
            (vec![3, 1], true),
        ]);
    }

    #[test]
    fn test_simple_var_muls() {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(60000);
        let ctx = z3::Context::new(&mut cfg);
        let _one = BV::from_u64(&ctx, 1, 8);
        let var0 = BV::new_const(&ctx, "var!00", 8);
        let var1 = BV::new_const(&ctx, "var!10", 8);

        let add = &var0 * &var1;
        let add2 = &add * &var1;
        let cst = add2._eq(&BV::from_u64(&ctx, 3, 8));
        jit_constraint_test(&function_name!(), vec![cst], &[
            (vec![3, 2], false),
            (vec![3, 1], true),
        ]);
    }
    #[test]
    fn test_simple_var_adds_sle() {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(60000);
        let ctx = z3::Context::new(&mut cfg);
        let _one = BV::from_u64(&ctx, 1, 8);
        let var0 = BV::new_const(&ctx, "var!00", 8);
        let var1 = BV::new_const(&ctx, "var!10", 8);

        let add = &var0 + &var1;
        let add2 = &add + &var0;
        let cst = add2.bvsle(&_one);
        jit_constraint_test(&function_name!(), vec![cst], &[
            (vec![1, 0], false),
            (vec![0, 1], true),
            (vec![1, 0xff], true),
            (vec![2, 0xff], false),
            (vec![0, 0xff], true),
        ]);
    }
    #[test]
    fn test_simple_var_adds_ule() {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(60000);
        let ctx = z3::Context::new(&mut cfg);
        let _one = BV::from_u64(&ctx, 1, 8);
        let var0 = BV::new_const(&ctx, "var!00", 8);
        let var1 = BV::new_const(&ctx, "var!10", 8);

        let add = &var0 + &var1;
        let add2 = &add + &var0;
        let cst = add2.bvule(&_one);
        jit_constraint_test(&function_name!(), vec![cst], &[
            (vec![1, 0], false),
            (vec![0, 1], true),
            (vec![1, 0xff], true),
            (vec![2, 0xff], false),
            (vec![0, 0xff], false),
        ]);
    }
    #[test]
    fn test_simple_var_extracts() {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(60000);
        let ctx = z3::Context::new(&mut cfg);
        let _one = BV::from_u64(&ctx, 1, 1);
        let var0 = BV::new_const(&ctx, "var!00", 8);

        let add = &var0.extract(0, 0);
        let cst = add._eq(&_one);
        jit_constraint_test(&function_name!(), vec![cst], &[
            (vec![0], false),
            (vec![1], true),
            (vec![2], false),
            (vec![3], true),
            (vec![0xfe], false),
            (vec![0xff], true),
        ]);
    }
    #[test]
    fn test_simple_var_concats() {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(60000);
        let ctx = z3::Context::new(&mut cfg);
        let var0 = BV::new_const(&ctx, "var!00", 8);
        let var1 = BV::new_const(&ctx, "var!10", 8);

        let concatted = var0.concat(&var1);
        let cst = concatted._eq(&BV::from_u64(&ctx, 0x1234, 16));
        jit_constraint_test(&function_name!(), vec![cst], &[
            (vec![0x12, 0x12], false),
            (vec![0x12, 0x34], true),
        ]);
        let concatted = concatted.concat(&var1);
        let cst = concatted._eq(&BV::from_u64(&ctx, 0x123434, 24));
        jit_constraint_test(&function_name!(), vec![cst], &[
            (vec![0x12, 0x12], false),
            (vec![0x12, 0x34], true),
        ]);
    }
    #[test]
    fn test_complicated_slowest_smt2() {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(60000);
        let ctx = z3::Context::new(&mut cfg);
        println!("cwd: {:?}", std::env::current_dir());
        let constraints = ctx.parse_file("slowest_50secs.sat.smt2").unwrap();
        let conjunction = constraints.iter().fold(Bool::from_bool(&ctx, true),
        |old, cur| {
            let boolcst = cur.as_bool().expect("Every constraint must be boolean");
            old & boolcst
        });
        let csts: Vec<Bool> = constraints
            .iter()
            .map(|x| x.as_bool().expect("constraints should always be of sort Bool!"))
            .collect();

        let ast_metadata = AstMetadata::from(&constraints[..]);

        println!("Getting model for the cst!");
        let model_true = get_bytes_model(&ctx, &ast_metadata, &csts[..]).expect("the constraint should be sat");
        println!("Got model {:?}, getting model for the negated constraint!", model_true);
        let mut vec2: Vec<Bool> = csts[0..1].iter().map(Bool::clone).collect();
        vec2.push(csts[2].not());
        vec2.extend(csts[3..].iter().map(Bool::clone));
        // let mut vec2: Vec<Bool> = vec![csts[0].not()];
        // vec2.extend(csts[1..].iter().map(Bool::clone));

        let mut model_neg = get_bytes_model(&ctx, &ast_metadata, &[conjunction.not()]).expect("the negation of the constraint should be sat");
        model_neg[1369] = 1;
        println!("Got negated model {:?}, testing!", model_neg);

        jit_constraint_test(&function_name!(), csts.clone(), &[
            (model_true.clone(), true),
            (model_neg.clone(), false),
        ]);
        println!("Finished testing!");

        let substitutions_true = ast_metadata.variable_to_byte
            .iter()
            .map(|(decl, idx)| {
                (decl.apply(&[]), BV::from_u64(&ctx, model_true[*idx] as u64, 8).into())
            })
            .collect::<Vec<_>>();
        let substitutions_neg = ast_metadata.variable_to_byte
            .iter()
            .map(|(decl, idx)| {
                (decl.apply(&[]), BV::from_u64(&ctx, model_neg[*idx] as u64, 8).into())
            })
            .collect::<Vec<_>>();

        let subs_pos = substitutions_true.iter()
            .map(|(var, ast)| (var, ast))
            .collect::<Vec<_>>();
        let subs_neg = substitutions_neg.iter()
            .map(|(var, ast)| (var, ast))
            .collect::<Vec<_>>();

        let (duration, result) = timeit!(conjunction.substitute(&subs_pos[..]).simplify().as_bool().unwrap());
        println!("cst.substitute(true_model).simplify() = {:?} took {:?}", result, duration);
        let (duration, result) = timeit!(conjunction.substitute(&subs_neg[..]).simplify().as_bool().unwrap());
        println!("cst.substitute(false_model).simplify() = {:?} took {:?}", result, duration);
    }
}