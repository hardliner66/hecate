use std::path::PathBuf;

use clap::Parser;
use fck::{
    lexer::lex,
    parser::{parse, AstNode},
};
use stringlit::s;

#[derive(Parser)]
struct Args {
    input: PathBuf,
    output: PathBuf,
}

const SC: &str = "R1";
const SC_ADDR: &str = "R2";
const SC_LEN: &str = "R3";
const Z: &str = "R0"; // Zero register
const DP: &str = "R10"; // Data pointer
const T1: &str = "R11"; // Temporary register 1
const T2: &str = "R12"; // Temporary register 2

pub fn ast_to_assembly(
    ast: &AstNode,
    asm: &mut Vec<String>,
    dp_offset: &mut isize,
    max_dp_offset_left: &mut isize,
    max_dp_offset_right: &mut isize,
) {
    match ast {
        AstNode::Sequence(nodes) => {
            for node in nodes {
                ast_to_assembly(
                    node,
                    asm,
                    dp_offset,
                    max_dp_offset_left,
                    max_dp_offset_right,
                );
            }
        }
        AstNode::Loop(nodes) => {
            let loop_start_label = format!("loop_start_{}", asm.len());
            let loop_end_label = format!("loop_end_{}", asm.len());

            asm.push(format!("{}:", loop_start_label));
            asm.push(format!("loadreg {T1}, {DP}"));
            asm.push(format!("cmp {T1}, {Z}"));
            asm.push(format!("je @{}", loop_end_label));
            for node in nodes {
                ast_to_assembly(
                    node,
                    asm,
                    dp_offset,
                    max_dp_offset_left,
                    max_dp_offset_right,
                );
            }
            asm.push(format!("jmp @{}", loop_start_label));
            asm.push(format!("{}:", loop_end_label));
        }
        AstNode::Clear => {
            asm.push(format!("load {DP}, 500"));
        }
        AstNode::Right(steps) => {
            asm.push(format!("add {DP}, {}", steps));
            *dp_offset += *steps as isize;
            if *dp_offset > *max_dp_offset_right {
                *max_dp_offset_right = *dp_offset;
            }
        }
        AstNode::Left(steps) => {
            asm.push(format!("sub {DP}, {}", steps));
            *dp_offset -= *steps as isize;
            if *dp_offset < *max_dp_offset_left {
                *max_dp_offset_left = *dp_offset;
            }
        }
        AstNode::Increment(value) => {
            asm.push(format!("loadreg {T2}, {DP}"));
            asm.push(format!("add {T2}, {}", value));
            asm.push(format!("storereg {DP}, {T2}"));
        }
        AstNode::Decrement(value) => {
            asm.push(format!("loadreg {T2}, {DP}"));
            asm.push(format!("sub {T2}, {}", value));
            asm.push(format!("storereg {DP}, {T2}"));
        }
        AstNode::Output => {
            asm.push(format!("load {SC}, {Z}"));
            asm.push(format!("load {SC_ADDR}, {DP}"));
            asm.push(format!("load {SC_LEN}, 1"));
            asm.push(s!("syscall"));
        }
        AstNode::Input => {
            panic!("Input not implemented");
        }
    }
}

fn main() -> anyhow::Result<()> {
    let Args { input, output } = Args::parse();

    let tokens = lex(&std::fs::read_to_string(input)?)?;
    let ast = parse(&tokens)?;
    let mut asm = Vec::new();
    let mut dp_offset = 0;
    let mut dp_offset_max_left = 0;
    let mut dp_offset_max_right = 0;
    asm.push(format!("load {DP}, 500"));
    ast_to_assembly(
        &ast,
        &mut asm,
        &mut dp_offset,
        &mut dp_offset_max_left,
        &mut dp_offset_max_right,
    );

    asm.push(s!("halt"));
    std::fs::write(output, asm.join("\n"))?;

    Ok(())
}
