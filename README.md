# Hecate Virtual Machine

This project provides an implementation of the Hecate virtual machine.

---

## Where does the name come from?

The project is named Hecate, inspired by the ancient Greek goddess associated with magic, crossroads, and guiding transformations.
Just as Hecate stood at the intersection of possibilities, this CPU project represents the convergence of computational logic,
optimization, and problem-solving, providing participants with a structured yet flexible virtual machine to explore and
optimize their solutions.

The name reflects both the mystical allure of programming challenges and the power of guiding complex tasks to completion.

---

## Features

- Bytecode Execution:
  - Supports a well-defined set of instructions for computation, memory operations, and control flow.
  - Instructions include:
    - Arithmetic operations (Add)
    - Memory access (LoadValue, LoadMemory, Store)
    - Stack manipulation (PushValue, PushReg, Pop)
    - Control flow (Call, Ret, Jmp)
    - Debugging (Inspect)
    - Halting (Halt)

- Performance Metrics:
  - Tracks the number of cycles executed.
  - Measures memory access performance using a multi-level cache scoring system (L1, L2, L3).

- Debugging and Verbose Mode:
  - **NOT YET IMPLEMENTED** Provides step-by-step debugging modes (StepOver, StepInto, StepOut) with support for code and data breakpoints.
  - Outputs detailed logs of executed instructions in verbose mode.

- Memory Model:
  - Flat memory structure with stack management.
  - Supports configurable memory size and register count.

---

## Getting Started

### Prerequisites

- Rust (for building the project)

---

## Usage

1. Clone the repository:
   git clone <repository-url>
   cd <repository-name>

2. Run the demo project:
   cargo run --release -- run-demo

---

## Bytecode Instruction Set
For simplicity, this implementation operates on 32 bit unsigned integers.
This might be changed later, but for now this should be enough.

| Opcode       | Mnemonic     | Description                                |
| ------------ | ------------ | ------------------------------------------ |
| `0x00`       | `Nop`        | No operation                               |
| `0x01`       | `LoadValue`  | Load immediate value into a register       |
| `0x02`       | `LoadMemory` | Load value from memory into a register     |
| `0x03`       | `Store`      | Store a register's value into memory       |
| `0x04`       | `PushValue`  | Push immediate value onto the stack        |
| `0x05`       | `PushReg`    | Push register value onto the stack         |
| `0x06`       | `Pop`        | Pop a value from the stack into a register |
| `0x10`       | `Add`        | Add two register values                    |
| `0xF0`       | `Call`       | Call a function                            |
| `0xF1`       | `Ret`        | Return from a function                     |
| `0xF2`       | `RetReg`     | Return and push a register value           |
| `0xF3`       | `Jmp`        | Jump to a specific address                 |
| `0xFE`       | `Inspect`    | Output the value of a memory address       |
| `0xFFFFFFFF` | `Halt`       | Halt the program                           |

---

## Performance Metrics

The CPU tracks performance using:
Cycles: Cumulative score based on average number of cycles (http://ithare.com/infographics-operation-costs-in-cpu-clock-cycles/):
- L1 Cache: 3 Cycles
- L2 Cache: 11 Cycles
- L3 Cache: 50 Cycles
- Main Memory: 125 Cycles

