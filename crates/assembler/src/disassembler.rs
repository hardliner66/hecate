use hecate_common::{Bytecode, BytecodeFile};
use num_traits::FromPrimitive;
use std::collections::HashSet;

fn operand_count(opcode: Bytecode) -> usize {
    use Bytecode::*;
    match opcode {
        Nop | Halt | Ret | Syscall => 0,

        Jmp | Je | Jne | Jg | Jge | Jl | Jle | Ja | Jae | Jb | Jbe | Jc | Jnc | Jo | Jno | Js
        | Jns | Jxcz | Call | Inspect => 1,

        _ => 2,
    }
}

fn gather_jump_targets(code: &[u32]) -> HashSet<u32> {
    let mut targets = HashSet::new();
    let mut i = 0;

    while i < code.len() {
        let opcode_val = code[i];
        i += 1;

        if let Some(opcode) = Bytecode::from_u32(opcode_val) {
            match opcode {
                Bytecode::Jmp
                | Bytecode::Je
                | Bytecode::Jne
                | Bytecode::Jg
                | Bytecode::Jge
                | Bytecode::Jl
                | Bytecode::Jle
                | Bytecode::Ja
                | Bytecode::Jae
                | Bytecode::Jb
                | Bytecode::Jbe
                | Bytecode::Jc
                | Bytecode::Jnc
                | Bytecode::Jo
                | Bytecode::Jno
                | Bytecode::Js
                | Bytecode::Jns
                | Bytecode::Jxcz
                | Bytecode::Call => {
                    if let Some(&addr) = code.get(i) {
                        targets.insert(addr);
                    }
                }
                _ => {}
            }

            i += operand_count(opcode);
        }
    }

    targets
}

pub fn disassemble_program(file: &BytecodeFile) -> Vec<String> {
    let code = &file.data;
    let jump_targets = gather_jump_targets(code);

    let mut lines = Vec::new();
    let mut i = 0;

    while i < code.len() {
        if let Some((label, _)) = file.header.labels.iter().find(|(_, a)| **a as usize == i) {
            lines.push(format!("{label}:"));
        } else if jump_targets.contains(&(i as u32)) {
            lines.push(format!("label_{}:", i));
        }

        let opcode_val = code[i];
        i += 1;

        // convert the numeric opcode to bytecode enum
        let opcode = match Bytecode::from_u32(opcode_val) {
            Some(op) => op,
            None => {
                // fallback in case of unknown opcode
                lines.push(format!("UNKNOWN_OPCODE {}", opcode_val));
                continue;
            }
        };

        match opcode {
            // 0-operand instructions
            Bytecode::Nop => {
                lines.push("nop".to_string());
            }
            Bytecode::Halt => {
                lines.push("halt".to_string());
            }
            Bytecode::Ret => {
                lines.push("ret".to_string());
            }
            Bytecode::Syscall => {
                lines.push("syscall".to_string());
            }

            // 2-operand instructions
            Bytecode::LoadMemory => {
                // load R{reg}, @{addr}
                let reg = code.get(i).copied().unwrap_or_default();
                let addr = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("load R{}, @{}", reg, addr));
            }
            Bytecode::LoadReg => {
                // load R{reg_dst}, R{reg_src}
                let dst = code.get(i).copied().unwrap_or_default();
                let src = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("load R{}, R{}", dst, src));
            }
            Bytecode::LoadValue => {
                // load R{reg}, {imm} (which can be integer bits or float bits)
                let reg = code.get(i).copied().unwrap_or_default();
                let imm = code.get(i + 1).copied().unwrap_or_default();
                i += 2;

                lines.push(format!("load R{}, {}", reg, imm));
            }
            Bytecode::Store => {
                // store @{addr}, R{reg}
                let addr = code.get(i).copied().unwrap_or_default();
                let reg = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("store @{}, R{}", addr, reg));
            }
            Bytecode::PushReg => {
                // push R{reg}
                let reg = code.get(i).copied().unwrap_or_default();
                i += 1;
                lines.push(format!("push R{}", reg));
            }
            Bytecode::PushValue => {
                // push {imm}
                let imm = code.get(i).copied().unwrap_or_default();
                i += 1;
                lines.push(format!("push {}", imm));
            }
            Bytecode::Pop => {
                // pop R{reg}
                let reg = code.get(i).copied().unwrap_or_default();
                i += 1;
                lines.push(format!("pop R{}", reg));
            }

            // Arithmetic ops (2 arguments)
            Bytecode::Add => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let r2 = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("add R{}, R{}", r1, r2));
            }
            Bytecode::AddValue => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let imm = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("add R{}, {}", r1, imm));
            }
            Bytecode::Sub => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let r2 = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("sub R{}, R{}", r1, r2));
            }
            Bytecode::SubValue => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let imm = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("sub R{}, {}", r1, imm));
            }
            Bytecode::Mul => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let r2 = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("mul R{}, R{}", r1, r2));
            }
            Bytecode::MulValue => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let imm = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("mul R{}, {}", r1, imm));
            }
            Bytecode::Div => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let r2 = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("div R{}, R{}", r1, r2));
            }
            Bytecode::DivValue => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let imm = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("div R{}, {}", r1, imm));
            }
            Bytecode::Cmp => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let r2 = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("cmp R{}, R{}", r1, r2));
            }
            Bytecode::CmpValue => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let imm = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("cmp R{}, {}", r1, imm));
            }

            // Branching ops
            Bytecode::Jmp
            | Bytecode::Je
            | Bytecode::Jne
            | Bytecode::Jg
            | Bytecode::Jge
            | Bytecode::Jl
            | Bytecode::Jle
            | Bytecode::Ja
            | Bytecode::Jae
            | Bytecode::Jb
            | Bytecode::Jbe
            | Bytecode::Jc
            | Bytecode::Jnc
            | Bytecode::Jo
            | Bytecode::Jno
            | Bytecode::Js
            | Bytecode::Jns
            | Bytecode::Jxcz
            | Bytecode::Call
            | Bytecode::Inspect => {
                let addr = code.get(i).copied().unwrap_or_default();
                i += 1;
                let mnemonic = match opcode {
                    Bytecode::Jmp => "jmp",
                    Bytecode::Je => "je",
                    Bytecode::Jne => "jne",
                    Bytecode::Jg => "jg",
                    Bytecode::Jge => "jge",
                    Bytecode::Jl => "jl",
                    Bytecode::Jle => "jle",
                    Bytecode::Ja => "ja",
                    Bytecode::Jae => "jae",
                    Bytecode::Jb => "jb",
                    Bytecode::Jbe => "jbe",
                    Bytecode::Jc => "jc",
                    Bytecode::Jnc => "jnc",
                    Bytecode::Jo => "jo",
                    Bytecode::Jno => "jno",
                    Bytecode::Js => "js",
                    Bytecode::Jns => "jns",
                    Bytecode::Jxcz => "jxcz",
                    Bytecode::Call => "call",
                    Bytecode::Inspect => "inspect",
                    _ => unreachable!(),
                };
                if let Some((label, _)) = file.header.labels.iter().find(|(_, a)| **a == addr) {
                    lines.push(format!("{} @{}", mnemonic, label));
                } else if jump_targets.contains(&addr) {
                    lines.push(format!("{} @label_{}", mnemonic, addr));
                } else {
                    lines.push(format!("{} @{}", mnemonic, addr));
                }
            }
            Bytecode::LoadByte => {
                let r = code.get(i).copied().unwrap_or_default();
                let addr = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("loadbyte R{}, @{}", r, addr));
            }
            Bytecode::StoreByte => {
                let addr = code.get(i).copied().unwrap_or_default();
                let r = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("storebyte @{}, R{}", addr, r));
            }

            // Float ops
            Bytecode::FAddValue => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let r2 = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("fadd R{}, R{}", r1, r2));
            }
            Bytecode::FAdd => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let bits = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                let float_val = f32::from_bits(bits);
                lines.push(format!("fadd R{}, {}", r1, float_val));
            }
            Bytecode::FSubValue => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let r2 = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("fsub R{}, R{}", r1, r2));
            }
            Bytecode::FSub => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let bits = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                let float_val = f32::from_bits(bits);
                lines.push(format!("fsub R{}, {}", r1, float_val));
            }
            Bytecode::FMulValue => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let r2 = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("fmul R{}, R{}", r1, r2));
            }
            Bytecode::FMul => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let bits = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                let float_val = f32::from_bits(bits);
                lines.push(format!("fmul R{}, {}", r1, float_val));
            }
            Bytecode::FDivValue => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let r2 = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("fdiv R{}, R{}", r1, r2));
            }
            Bytecode::FDiv => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let bits = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                let float_val = f32::from_bits(bits);
                lines.push(format!("fdiv R{}, {}", r1, float_val));
            }
            Bytecode::FCmpValue => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let r2 = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                lines.push(format!("fcmp R{}, R{}", r1, r2));
            }
            Bytecode::FCmp => {
                let r1 = code.get(i).copied().unwrap_or_default();
                let bits = code.get(i + 1).copied().unwrap_or_default();
                i += 2;
                let float_val = f32::from_bits(bits);
                lines.push(format!("fcmp R{}, {}", r1, float_val));
            }
        }
    }

    lines
        .into_iter()
        .map(|l| {
            if l.trim().ends_with(":") {
                l
            } else {
                format!("  {l}")
            }
        })
        .collect()
}
