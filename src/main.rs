use std::{io::BufWriter, path::PathBuf};

use byteorder::{ReadBytesExt, WriteBytesExt};
use clap::{Parser, Subcommand};
use common::{CpuTrait, RunMode};
use native::NativeCpu;
use std::io::BufReader;

mod native;

#[derive(Parser)]
struct Args {
    #[arg(short, long, global = true)]
    verbose: bool,
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    Run { path: PathBuf },
    RunDemo,
    WriteDemo { path: PathBuf },
}

use common::Bytecode::*;

const DEMO2: &[(i32, &[u32])] = &[(
    0,
    &[
        1, 1, 5, 3072, 0, 0, 3074, 25, 18, 1, 0, 1, 2, 1, 17, 0, 2, 1, 2, 0, 3072, 0, 2, 3076, 8,
        5, 1, 6, 0, 4294967295,
    ],
)];

const DEMO: &[(i32, &[u32])] = &[
    // MAIN (starting at 0)
    (
        0,
        &[
            /*0:*/ LoadValue as u32,
            /*1:*/ 0, // R0
            /*2:*/ 5, // R0 = 5
            /*3:*/ Call as u32,
            /*4:*/ 40, // Call factorial at address 40
            /*7:*/ Store as u32,
            /*8:*/ 100, // Store at memory address 100
            /*9:*/ 1, // from R1
            /*10:*/ Inspect as u32,
            /*11:*/ 100, // Inspect memory address 100
            /*12:*/ Halt as u32, // Halt
        ],
    ),
    (
        40,
        &[
            // FACTORIAL function at address 40
            /*40:*/ LoadValue as u32,
            /*41:*/ 1, // R1
            /*42:*/ 1, // R1 = 1
            /*43:*/ Cmp as u32,
            /*44:*/ 0, // Compare R0
            /*45:*/ 1, // with R1
            /*46:*/ Jle as u32,
            /*47:*/ 80, // If R0 <= 1, jump to FACT_BASE at 80
            // Not base case:
            // Push original R0
            // R0 = R0 - R1 (subtract 1 from R0)
            /*48:*/
            Sub as u32,
            /*49:*/ 0, // R0
            /*50:*/ 1, // R1
            // Call factorial again (recursive)
            /*51:*/ Call as u32,
            /*52:*/ 40, // Call factorial at address 40
            // Upon return, pop result into R3
            /*53:*/ Pop as u32,
            /*54:*/ 3, // R3 = factorial(R0-1)
            // Pop original n back into R0
            /*55:*/ Pop as u32,
            /*56:*/ 0,
            // R0 = R0 * R3
            /*57:*/ Mul as u32,
            /*58:*/ 0, // R0
            /*59:*/ 3, // R3
            // Ret
            /*60:*/ Ret as u32,
        ],
    ),
    (
        80,
        &[
            // FACT_BASE at 80: base case returns 1
            /*80:*/ LoadValue as u32,
            /*81:*/ 0, // R0
            /*82:*/ 1, // R0 = 1
            /*83:*/ Ret as u32,
            /*84:*/ 0,
        ],
    ),
];

fn demo_to_bytecode(demo: &[(i32, &[u32])]) -> Vec<u32> {
    let mut bytecode = Vec::new();
    for (start, code) in demo {
        while bytecode.len() < *start as usize {
            bytecode.push(Nop as u32);
        }
        bytecode.extend_from_slice(code);
    }
    bytecode
}

fn run(memory: &[u32], verbose: bool) -> anyhow::Result<()> {
    let mut cpu = NativeCpu::new(1024 * 1024, 6);
    cpu.set_verbose(verbose);

    cpu.load_memory(0, memory);

    let stats = cpu.execute(RunMode::Run)?;

    println!();
    println!(" ========== STATS ===========");
    println!();

    println!("{:#?}", stats);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let Args { action, verbose } = Args::parse();

    match action {
        Action::RunDemo => {
            run(&demo_to_bytecode(DEMO2), verbose)?;
        }
        Action::WriteDemo { path } => {
            let file = std::fs::File::create(path)?;
            let mut writer = BufWriter::new(file);
            for v in demo_to_bytecode(DEMO) {
                writer.write_u32::<byteorder::LittleEndian>(v)?;
            }
        }

        Action::Run { path } => {
            let memory = std::fs::File::open(path)
                .map(|file| {
                    let mut reader = BufReader::new(file);
                    let mut values = Vec::new();
                    while let Ok(v) = reader.read_u32::<byteorder::LittleEndian>() {
                        values.push(v);
                    }
                    values
                })
                .unwrap();

            run(&memory, verbose)?;
        }
    }

    Ok(())
}
