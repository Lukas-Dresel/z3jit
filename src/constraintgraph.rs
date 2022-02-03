use std::fmt::{Display, Debug};
use std::collections::HashMap;

use itertools::Itertools;

use petgraph::Undirected;
use petgraph::graph::{Node, NodeIndex};
use petgraph::{Graph, dot::{Config, Dot}};
use z3::ast::Bool;
use z3::{ast::{BV, Ast, Dynamic}, SatResult};

trait TopologicalSort<'ctx> {
    fn topologically_sorted_nodes(&self) -> Vec<Dynamic<'ctx>>;
}
impl<'ctx> TopologicalSort<'ctx> for Dynamic<'ctx> {
    fn topologically_sorted_nodes(&self) -> Vec<Dynamic<'ctx>> {
        let mut result_vec: Vec<Dynamic> = vec![];
        for child in self.children() {
            result_vec.append(&mut child.topologically_sorted_nodes());
        }
        result_vec.push(self.clone());
        let result = result_vec.into_iter().unique().collect();
        println!("Toposort({:?}) = {:?}", self, result);
        result
    }
}

#[derive(Default, Debug, Clone)]
struct ConstraintGraph<'ctx> {
    ast_to_idx: HashMap<Dynamic<'ctx>, NodeIndex<usize>>,
    graph: Graph<Dynamic<'ctx>, usize, Undirected, usize>
}

impl<'ctx> ConstraintGraph<'ctx> {
    fn new() -> ConstraintGraph<'ctx> {
        Default::default()
    }
    fn from_constraint(cst: Bool<'ctx>) -> ConstraintGraph<'ctx> {
        let mut graph: ConstraintGraph = Default::default();

        let cst : Dynamic = cst.into();

        for node in cst.topologically_sorted_nodes() {
            println!("Inserting {:?}", node);
            assert!(!graph.ast_to_idx.contains_key(&node));
            let my_graph_idx: NodeIndex<usize> = graph.graph.add_node(node.clone());

            for (child_idx, child) in node.children().into_iter().enumerate() {
                let child_graph_idx = *graph.ast_to_idx
                    .get(&child)
                    .expect("How are children not yet in there, if this is supposed to be topologically sorted?");
                graph.graph.update_edge(my_graph_idx, child_graph_idx, child_idx);
            }
            let node_repr = format!("{:?}", node);
            graph.ast_to_idx.insert(node, my_graph_idx);
            println!("graph after inserting {:?}: {:?}", node_repr, graph);
        }
        graph
    }
    fn dump(&self, filename: &str) {
        assert!(filename.ends_with(".dot"));

        let content = Dot::with_config(&self.graph, &[]);
        std::fs::write(filename, format!("{:?}", content)).expect("why can't i write");
    }
}






#[cfg(test)]
mod tests {
    use petgraph::dot::{Config, Dot};
    use z3::{ast::{BV, Ast, Dynamic}, SatResult};

    use crate::ast_verbose_print::VerbosePrint;

    use super::ConstraintGraph;

    #[test]
    fn test_empty() {
        let graph = ConstraintGraph::new();
        graph.dump("empty.dot");
    }

    #[test]
    fn it_works() {
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
        let _dyn : Dynamic = cst.clone().into();
        _dyn.print_verbose();

        let mut graph = petgraph::Graph::<Option<Dynamic>, usize>::new();
        graph.add_node(Some(_one.into()));
        graph.add_node(Some(var0.into()));
        graph.add_node(Some(var1.into()));
        graph.add_node(Some(add.into()));
        graph.add_node(Some(add2.into()));
        graph.add_node(Some(cst.clone().into()));
        graph.extend_with_edges(&[
            (3, 1), (3, 2), (4, 3), (4, 1), (5, 4), (5, 0)
        ]);
        let dot_repr = format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));
        std::fs::write("expected.dot", dot_repr).expect("write you ass");

        let graph = ConstraintGraph::from_constraint(cst);
        graph.dump("actual.dot");
    }
}
