use std::collections::HashMap;
use z3::{FuncDecl, DeclKind};
use z3::ast::{Ast, Dynamic};
use z3::ast_visitor::{AstVisitor, AstTraversal};


fn var_name_to_index(name: &str) -> usize {
    let ind = name.find('!').expect(&format!("Invalid variable name!!! {:?}", name));
    let (_pre, post) = name.split_at(ind+1);
    // assert!(pre == "k", "Variable name must start with k: {:?}", name);
    let byte_idx_mult = usize::from_str_radix(post, 10)
        .expect(&format!("Variable index cannot be parsed: {:?}, post={:?}", name, post));
    assert!(byte_idx_mult % 10 == 0, "Parsed out invalid index from {:?}: index={:?}, should be a multiple of 10!", name, byte_idx_mult);
    return byte_idx_mult / 10;
}

#[derive(Debug, Default, Clone)]
pub struct AstMetadata<'ctx> {
    pub variable_to_byte: HashMap<FuncDecl<'ctx>, usize>,
    pub ast_depths: HashMap<Dynamic<'ctx>, usize>,
    pub max_byte_index : usize,
}

impl<'ctx> AstMetadata<'ctx> {
    pub fn from(asts: &[Dynamic<'ctx>]) -> AstMetadata<'ctx> {
        let mut meta : AstMetadata = Default::default();
        for a in asts {
            meta.track_ast(a);
        }
        meta
    }
    pub fn track_ast(&mut self, ast: &Dynamic<'ctx>) {
        ast.accept_depth_first(self, true, Default::default());
    }

    fn visit_generic(&mut self, ast: &Dynamic<'ctx>) {
        let depth = ast.children().iter().map(
            |child| self.ast_depths.get(child).unwrap()
        ).max().unwrap_or(&0) + 1;
        self.ast_depths.insert(ast.clone(), depth);
    }
}

impl<'ctx> AstVisitor<'ctx> for AstMetadata<'ctx> {
    fn visit_Numeral(&mut self, ast: &Dynamic<'ctx>) {
        self.visit_generic(ast)
    }
    fn visit_FuncDecl(&mut self, ast: &Dynamic<'ctx>) {
        self.visit_generic(ast)
    }
    fn visit_Quantifier(&mut self, ast: &Dynamic<'ctx>) {
        self.visit_generic(ast)
    }
    fn visit_Sort(&mut self, ast: &Dynamic<'ctx>) {
        self.visit_generic(ast)
    }
    fn visit_Unknown(&mut self, ast: &Dynamic<'ctx>) {
        self.visit_generic(ast)
    }
    fn visit_Var(&mut self, ast: &Dynamic<'ctx>) {
        self.visit_generic(ast)
    }

    fn visit_App(&mut self, ast: &Dynamic<'ctx>) {
        self.visit_generic(ast);

        let decl = ast.decl();
        if decl.arity() != 0 {
            return
        }
        if decl.kind() != DeclKind::UNINTERPRETED {
            return;
        }
        let idx = var_name_to_index(&decl.name());
        self.variable_to_byte.insert(decl, idx);
        self.max_byte_index = self.max_byte_index.max(idx+1);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use z3::{ast::{BV, Ast, Dynamic}};
    use crate::{ast_verbose_print::VerbosePrint, ast_metadata::AstMetadata};

    #[test]
    fn it_works() {
        let mut cfg = z3::Config::new();
        cfg.set_model_generation(true);
        cfg.set_timeout_msec(60000);
        let ctx = z3::Context::new(&mut cfg);
        let _one = BV::from_u64(&ctx, 1, 8);
        let var0 = BV::new_const(&ctx, "var!0", 8);
        let var1 = BV::new_const(&ctx, "var!10", 8);

        let add = &var0 + &var1;
        let add2 = &add + &var0;
        let cst = add2._eq(&_one);
        let _dyn : Dynamic = cst.clone().into();
        _dyn.print_verbose();

        let expected = [(var0.decl(), 0), (var1.decl(), 1)].into_iter().collect::<HashMap<_, _>>();
        let meta = AstMetadata::from(&[cst.into()]);
        println!("Expected: {:?}, actual: {:?}", expected, meta.variable_to_byte);
        assert!(expected == meta.variable_to_byte);
    }
}
