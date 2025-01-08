use std::{collections::BTreeMap, ops::Range};

use crate::{Bytecode, CpuStats, ExecutionError, RunMode};
use num_traits::FromPrimitive;

const L1_LATENCY: usize = 3;
const L2_LATENCY: usize = 11;
const L3_LATENCY: usize = 50;
const MEMORY_LATENCY: usize = 125;
const AVERAGE_STORE_LATENCY: usize = 1;

const LINE_SIZE: u32 = 64;
const L1_SETS: usize = 64;
const L1_WAYS: usize = 4;
const L2_SETS: usize = 256;
const L2_WAYS: usize = 8;
const L3_SETS: usize = 1024;
const L3_WAYS: usize = 16;

const REGISTER_WIDTH: u32 = 32; // Example: 8, 16, 32, or 64, etc.

// Calculate how many bits we need to mask off for shifting
// e.g. 32.trailing_zeros() = 5; 64.trailing_zeros() = 6
const SHIFT_BITS: u32 = REGISTER_WIDTH.trailing_zeros();

// Then your mask is (1 << SHIFT_BITS) - 1
const SHIFT_MASK: u32 = (1 << SHIFT_BITS) - 1;

#[derive(Debug, Default, Clone)]
struct CacheLine {
    valid: bool,
    tag: u32,
}

#[derive(Debug)]
struct Cache {
    sets: Vec<Vec<CacheLine>>,
    sets_mask: u32,
    line_offset_bits: u32,
    tag_shift: u32,
    latency: usize,
}

impl Cache {
    fn new(num_sets: usize, ways: usize, line_size: u32, latency: usize) -> Self {
        let line_offset_bits = line_size.trailing_zeros();
        let set_count = num_sets as u32;
        let set_index_bits = set_count.trailing_zeros();

        let sets = vec![vec![CacheLine::default(); ways]; num_sets];

        Self {
            sets,
            sets_mask: set_count - 1,
            line_offset_bits,
            tag_shift: line_offset_bits + set_index_bits,
            latency,
        }
    }

    fn access(&mut self, address: u32, tag: u32) -> Option<usize> {
        let set_index = ((address >> self.line_offset_bits) & self.sets_mask) as usize;
        let set = &mut self.sets[set_index];

        if let Some(pos) = set.iter().position(|line| line.valid && line.tag == tag) {
            // Hit. Move line to front for LRU policy
            let line = set.remove(pos);
            set.insert(0, line);
            return Some(self.latency);
        }

        None
    }

    fn insert_line(&mut self, address: u32, tag: u32) {
        let set_index = ((address >> self.line_offset_bits) & self.sets_mask) as usize;
        let set = &mut self.sets[set_index];

        // Evict LRU (end of vector)
        let mut evict_line = set.pop().unwrap();
        evict_line.valid = true;
        evict_line.tag = tag;

        // Insert as MRU
        set.insert(0, evict_line);
    }

    fn line_tag(&self, address: u32) -> u32 {
        address >> self.tag_shift
    }
}

#[derive(Debug, PartialEq, Eq)]
enum MemoryAccessDirection {
    LoadData,
    LoadInstruction,
    Prefetch,
    Store,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Flags {
    pub zero: bool,
    pub carry: bool,
    pub sign: bool,
    pub overflow: bool,
}

pub trait HostIO: std::fmt::Debug {
    fn syscall(&mut self, code: u32, cpu: &mut NativeCpu<Self>) -> Result<usize, ExecutionError>
    where
        Self: Sized;
}

#[derive(Debug)]
pub struct NullHostIO;

impl HostIO for NullHostIO {
    fn syscall(
        &mut self,
        _code: u32,
        _cpu: &mut NativeCpu<NullHostIO>,
    ) -> Result<usize, ExecutionError> {
        // Default: do nothing
        Ok(1250)
    }
}

#[derive(Debug)]
pub struct NativeCpu<IO: HostIO> {
    memory: Vec<u32>,
    registers: Vec<u32>,
    instruction_pointer: u32,
    stack_pointer: u32,
    stats: CpuStats,
    flags: Flags,
    l1i: Cache,
    l1d: Cache,
    l2: Cache,
    l3: Cache,

    protected_memory: Range<u32>,

    last_load_addresses: Vec<u32>,
    stable_stride: Option<i32>,
    pub host_io: Option<IO>,
    halted: bool,
    verbose: bool,
    print_memory_access: bool,
    addresses_as_integers: bool,
}

impl<IO: HostIO> NativeCpu<IO> {
    pub fn set_addresses_as_integers(&mut self, addresses_as_integers: bool) {
        self.addresses_as_integers = addresses_as_integers;
    }

    pub fn print_state(&self) {
        println!();
        println!("========== VM STATE ===========");
        println!();
        println!("IP: {}", self.instruction_pointer);
        println!("SP: {}", self.stack_pointer);
        println!("Flags: {:#?}", self.flags);
        println!(
            "Registers: {:#?}",
            self.registers
                .iter()
                .enumerate()
                .collect::<BTreeMap<_, _>>()
        );
        println!(
            "Memory: {:#?}",
            self.memory.iter().enumerate().collect::<BTreeMap<_, _>>()
        );
    }

    pub fn set_halted(&mut self, halted: bool) {
        self.halted = halted;
    }

    #[allow(unused)]
    pub fn get_halted(&self) -> bool {
        self.halted
    }

    pub fn set_print_memory_access(&mut self, print_memory_access: bool) {
        self.print_memory_access = print_memory_access;
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    pub fn set_entrypoint(&mut self, entrypoint: u32) {
        self.instruction_pointer = entrypoint;
    }

    pub fn protect(&mut self, range: Range<u32>) {
        self.protected_memory = range;
    }

    pub fn load_protected_memory(&mut self, address: u32, memory: &[u32]) {
        let len = self.memory.len().min(memory.len());
        self.memory[address as usize..address as usize + len].copy_from_slice(&memory[..len]);
        self.protect(address..address + memory.len() as u32);
    }

    pub fn load_memory(&mut self, address: u32, memory: &[u32]) {
        let len = self.memory.len().min(memory.len());
        self.memory[address as usize..address as usize + len].copy_from_slice(&memory[..len]);
    }

    pub fn execute(&mut self, run_mode: RunMode) -> Result<CpuStats, ExecutionError> {
        match run_mode {
            RunMode::Run => self.run(-1),
            RunMode::RunFor(cycles) => self.run(cycles),
            _ => Err(ExecutionError::NotImplemented),
        }
    }

    pub fn get_registers(&self) -> &[u32] {
        &self.registers
    }

    pub fn get_mut_registers(&mut self) -> &mut [u32] {
        &mut self.registers
    }

    pub fn get_memory(&self) -> &[u32] {
        &self.memory
    }

    pub fn new(memory_size: u32, registers: u8, host_io: IO) -> Self {
        Self {
            memory: vec![0; memory_size as usize],
            registers: vec![0; registers as usize],
            instruction_pointer: 0,
            stack_pointer: memory_size - 1,
            verbose: false,
            stats: CpuStats::default(),
            flags: Flags::default(),
            l1i: Cache::new(L1_SETS, L1_WAYS, LINE_SIZE, L1_LATENCY),
            l1d: Cache::new(L1_SETS, L1_WAYS, LINE_SIZE, L1_LATENCY),
            l2: Cache::new(L2_SETS, L2_WAYS, LINE_SIZE, L2_LATENCY),
            l3: Cache::new(L3_SETS, L3_WAYS, LINE_SIZE, L3_LATENCY),

            protected_memory: 0..0,

            last_load_addresses: Vec::new(),
            stable_stride: None,
            host_io: Some(host_io),
            halted: false,
            print_memory_access: false,
            addresses_as_integers: false,
        }
    }

    /// Perform a logical left shift by `count` bits, returning (new_value, carry_bit).
    fn shift_left(&self, value: u32, count: u32) -> (u32, bool) {
        let shift = count & SHIFT_MASK;
        if shift == 0 {
            return (value, false);
        }
        // The carry bit is the leftmost bit that gets shifted out:
        let carry_out = ((value >> (32 - shift)) & 1) == 1;
        let result = value.wrapping_shl(shift);
        (result, carry_out)
    }

    /// Perform a logical right shift by `count` bits, returning (new_value, carry_bit).
    fn shift_right(&self, value: u32, count: u32) -> (u32, bool) {
        let shift = count & 31;
        if shift == 0 {
            return (value, false);
        }
        // The carry bit is the rightmost bit that gets shifted out:
        let carry_out = ((value >> (shift - 1)) & 1) == 1;
        let result = value.wrapping_shr(shift);
        (result, carry_out)
    }

    /// Update flags after a logical left shift.
    /// If shift_count == 1, we follow x86-like semantics: Overflow is set if MSB changed.
    /// Otherwise, we clear overflow.
    fn update_flags_shl(&mut self, original: u32, result: u32, carry_out: bool, shift_count: u32) {
        self.flags.zero = result == 0;
        self.flags.sign = (result as i32) < 0;
        self.flags.carry = carry_out;

        // Overflow: if shifting by 1, check if the sign bit changed
        if shift_count == 1 {
            let msb_original = (original >> 31) & 1;
            let msb_result = (result >> 31) & 1;
            self.flags.overflow = msb_original != msb_result;
        } else {
            self.flags.overflow = false;
        }
    }

    /// Update flags after a logical right shift.
    /// Typically, there's no concept of overflow for logical right shift, so we set OF=0.
    fn update_flags_shr(&mut self, result: u32, carry_out: bool) {
        self.flags.zero = result == 0;
        self.flags.sign = (result as i32) < 0;
        self.flags.carry = carry_out;
        self.flags.overflow = false;
    }

    fn access(&mut self, address: u32, direction: MemoryAccessDirection) -> usize {
        self.stats.memory_access_count += 1;
        let l1_cache = if direction == MemoryAccessDirection::LoadInstruction {
            &mut self.l1i
        } else {
            &mut self.l1d
        };
        let tag_l1 = l1_cache.line_tag(address);

        // Check L1
        if let Some(l1_lat) = l1_cache.access(address, tag_l1) {
            if self.print_memory_access && direction != MemoryAccessDirection::Prefetch {
                println!(" (L1 HIT)");
            }

            if direction == MemoryAccessDirection::LoadInstruction {
                self.stats.cache_hits.l1i += 1;
            } else {
                self.stats.cache_hits.l1d += 1;
            }
            // L1 hit
            return match direction {
                MemoryAccessDirection::LoadData
                | MemoryAccessDirection::LoadInstruction
                | MemoryAccessDirection::Prefetch => l1_lat,
                MemoryAccessDirection::Store => AVERAGE_STORE_LATENCY,
            };
        }

        let tag_l2 = self.l2.line_tag(address);

        // L1 miss, check L2
        if let Some(l2_lat) = self.l2.access(address, tag_l2) {
            if self.print_memory_access && direction != MemoryAccessDirection::Prefetch {
                println!(" (L2 HIT)");
            }
            self.stats.cache_hits.l2 += 1;
            // L2 hit: bring line to L1
            l1_cache.insert_line(address, tag_l1);

            return match direction {
                MemoryAccessDirection::LoadData
                | MemoryAccessDirection::LoadInstruction
                | MemoryAccessDirection::Prefetch => l2_lat,
                MemoryAccessDirection::Store => AVERAGE_STORE_LATENCY,
            };
        }

        let tag_l3 = self.l3.line_tag(address);

        // L2 miss, check L3
        if let Some(l3_lat) = self.l3.access(address, tag_l3) {
            if self.print_memory_access && direction != MemoryAccessDirection::Prefetch {
                println!(" (L3 HIT)");
            }
            self.stats.cache_hits.l3 += 1;
            // L3 hit: bring line into L2 and L1
            self.l2.insert_line(address, tag_l2);
            l1_cache.insert_line(address, tag_l1);

            return match direction {
                MemoryAccessDirection::LoadData
                | MemoryAccessDirection::LoadInstruction
                | MemoryAccessDirection::Prefetch => l3_lat,
                MemoryAccessDirection::Store => AVERAGE_STORE_LATENCY,
            };
        }

        if self.print_memory_access && direction != MemoryAccessDirection::Prefetch {
            println!(" (CACHE MISS)");
        }

        // Miss in all caches, fetch from memory
        // Insert line into L3, L2, L1
        self.l3.insert_line(address, tag_l3);
        self.l2.insert_line(address, tag_l2);
        l1_cache.insert_line(address, tag_l1);

        match direction {
            MemoryAccessDirection::LoadData
            | MemoryAccessDirection::LoadInstruction
            | MemoryAccessDirection::Prefetch => MEMORY_LATENCY,
            MemoryAccessDirection::Store => AVERAGE_STORE_LATENCY,
        }
    }

    fn detect_stride(&mut self) {
        // We want to detect a stable stride from recent load addresses
        // Let's say we need at least 4 load addresses to guess a pattern.
        if self.last_load_addresses.len() < 4 {
            self.stable_stride = None;
            return;
        }

        // Consider last 4 addresses to detect a pattern
        let len = self.last_load_addresses.len();
        let a1 = self.last_load_addresses[len - 4];
        let a2 = self.last_load_addresses[len - 3];
        let a3 = self.last_load_addresses[len - 2];
        let a4 = self.last_load_addresses[len - 1];

        let d1 = a2 as i32 - a1 as i32;
        let d2 = a3 as i32 - a2 as i32;
        let d3 = a4 as i32 - a3 as i32;

        // Check if strides are consistent
        if d1 == d2 && d2 == d3 {
            // We have a stable stride
            self.stable_stride = Some(d1);
        } else {
            self.stable_stride = None;
        }
    }

    fn prefetch_instructions(&mut self, current_address: u32) {
        let words_per_line = (LINE_SIZE / 4) as i32;

        for i in 0..10 {
            let next_line_address = ((current_address as i32) + (words_per_line * i)) as u32;

            // We ignore the cost here, as this is normally done by the hardware in the background
            _ = self.access(next_line_address, MemoryAccessDirection::Prefetch);
        }
    }

    fn prefetch_lines(&mut self, current_address: u32) {
        // If no stable stride, no prefetch
        let stride = match self.stable_stride {
            Some(s) => s,
            None => return,
        };

        if stride == 0 {
            return; // no stride to prefetch
        }

        // Ensure stride moves forward. If stride is negative, we could still prefetch backwards,
        // but we'll just handle positive stride for simplicity.
        if stride > 0 {
            let words_per_line = (LINE_SIZE / 4) as i32;

            let next_line_address = ((current_address as i32) + stride) as u32;
            let second_line_address = ((current_address as i32) + stride + words_per_line) as u32;

            // Prefetch both lines. Treat them as loads.
            // We ignore the cost here, as this is normally done by the hardware in the background
            _ = self.access(next_line_address, MemoryAccessDirection::Prefetch);
            _ = self.access(second_line_address, MemoryAccessDirection::Prefetch);
        }
    }

    fn update_load_history(&mut self, address: u32) {
        self.last_load_addresses.push(address);
        if self.last_load_addresses.len() > 16 {
            self.last_load_addresses.remove(0);
        }
        self.detect_stride();
    }

    fn read_instruction(&mut self, address: u32) -> Result<Bytecode, ExecutionError> {
        self.valid_address(address)?;
        if self.print_memory_access {
            if self.addresses_as_integers {
                print!("READ INSTR @{}", address);
            } else {
                print!("READ INSTR @{:#02x}", address);
            }
        }
        let cost = self.access(address, MemoryAccessDirection::LoadInstruction);
        self.stats.cycles += cost;

        self.prefetch_instructions(address);

        let value = self.memory[address as usize];

        match Bytecode::from_u32(value) {
            Some(instr) => {
                self.instruction_pointer += 1;
                Ok(instr)
            }
            None => Err(ExecutionError::InvalidOpcode(
                value,
                self.instruction_pointer,
            )),
        }
    }

    fn read_operand(&mut self, address: u32) -> Result<u32, ExecutionError> {
        let result = self.read_memory(address)?;
        self.instruction_pointer += 1;
        Ok(result)
    }

    fn read_memory(&mut self, address: u32) -> Result<u32, ExecutionError> {
        self.valid_address(address)?;
        if self.print_memory_access {
            if self.addresses_as_integers {
                print!("READ @{}", address);
            } else {
                print!("READ @{:#02x}", address);
            }
        }
        let cost = self.access(address, MemoryAccessDirection::LoadData);
        self.stats.cycles += cost;

        self.update_load_history(address);

        self.prefetch_lines(address);

        Ok(self.memory[address as usize])
    }

    fn write_memory(&mut self, address: u32, value: u32) -> Result<(), ExecutionError> {
        self.valid_address(address)?;
        if self.protected_memory.contains(&address) {
            return Err(ExecutionError::WriteProtectedMemory(address));
        }
        if self.print_memory_access {
            if self.addresses_as_integers {
                print!("WRITE @{}, {}", address, value);
            } else {
                print!("WRITE @{:#02x}, {}", address, value);
            }
        }
        let cost = self.access(address, MemoryAccessDirection::Store);
        self.stats.cycles += cost;
        self.memory[address as usize] = value;
        Ok(())
    }

    /// Update flags after an AND operation.
    ///
    /// Typical behavior:
    /// - ZERO = (result == 0)
    /// - SIGN = (result's most significant bit set)
    /// - CARRY = false
    /// - OVERFLOW = false
    fn update_flags_and(&mut self, a: u32, b: u32, result: u32) {
        self.flags.zero = result == 0;
        self.flags.sign = (result as i32) < 0;
        self.flags.carry = false;
        self.flags.overflow = false;

        if self.verbose {
            println!(
                "FLAGS (AND): a=0x{:08X}, b=0x{:08X}, result=0x{:08X}, zero={}, sign={}, carry={}, overflow={}",
                a, b, result, self.flags.zero, self.flags.sign, self.flags.carry, self.flags.overflow
            );
        }
    }

    /// Update flags after an OR operation.
    ///
    /// Typical behavior:
    /// - ZERO = (result == 0)
    /// - SIGN = (result's most significant bit set)
    /// - CARRY = false
    /// - OVERFLOW = false
    fn update_flags_or(&mut self, a: u32, b: u32, result: u32) {
        self.flags.zero = result == 0;
        self.flags.sign = (result as i32) < 0;
        self.flags.carry = false;
        self.flags.overflow = false;

        if self.verbose {
            println!(
                "FLAGS (OR): a=0x{:08X}, b=0x{:08X}, result=0x{:08X}, zero={}, sign={}, carry={}, overflow={}",
                a, b, result, self.flags.zero, self.flags.sign, self.flags.carry, self.flags.overflow
            );
        }
    }

    /// Update flags after an XOR operation.
    ///
    /// Typical behavior:
    /// - ZERO = (result == 0)
    /// - SIGN = (result's most significant bit set)
    /// - CARRY = false
    /// - OVERFLOW = false
    fn update_flags_xor(&mut self, a: u32, b: u32, result: u32) {
        self.flags.zero = result == 0;
        self.flags.sign = (result as i32) < 0;
        self.flags.carry = false;
        self.flags.overflow = false;

        if self.verbose {
            println!(
                "FLAGS (XOR): a=0x{:08X}, b=0x{:08X}, result=0x{:08X}, zero={}, sign={}, carry={}, overflow={}",
                a, b, result, self.flags.zero, self.flags.sign, self.flags.carry, self.flags.overflow
            );
        }
    }

    /// Update flags after a NOT operation.
    ///
    /// Typical behavior:
    /// - ZERO = (result == 0)
    /// - SIGN = (result's most significant bit set)
    /// - CARRY = false
    /// - OVERFLOW = false
    fn update_flags_not(&mut self, a: u32, result: u32) {
        self.flags.zero = result == 0;
        self.flags.sign = (result as i32) < 0;
        self.flags.carry = false;
        self.flags.overflow = false;

        if self.verbose {
            println!(
                "FLAGS (NOT): a=0x{:08X}, result=0x{:08X}, zero={}, sign={}, carry={}, overflow={}",
                a, result, self.flags.zero, self.flags.sign, self.flags.carry, self.flags.overflow
            );
        }
    }

    fn update_flags_mul(&mut self, _a: u32, _b: u32, result: u64) {
        // The final value is the lower 32 bits
        let low_result = result as u32;

        // Zero flag
        self.flags.zero = low_result == 0;

        // Sign flag (check MSB of low_result)
        self.flags.sign = (low_result as i32) < 0;

        // Carry and Overflow:
        // If the upper 32 bits of result are not zero, it means the multiplication overflowed the 32-bit range.
        let high_result = (result >> 32) as u32;
        let overflow_occurred = high_result != 0;
        self.flags.carry = overflow_occurred;
        self.flags.overflow = overflow_occurred;
    }

    fn update_flags_div(&mut self, _a: u32, _b: u32, result: u32) {
        self.flags.carry = false;
        self.flags.overflow = false;

        // Zero flag: set if result is zero
        self.flags.zero = result == 0;

        // Sign flag
        self.flags.sign = (result as i32) < 0;
    }

    fn should_jump(&self, instr: Bytecode) -> bool {
        let f = &self.flags;
        match instr {
            // Signed conditions
            Bytecode::Je => f.zero,
            Bytecode::Jne => !f.zero,
            Bytecode::Jg => !f.zero && (f.sign == f.overflow),
            Bytecode::Jge => f.sign == f.overflow,
            Bytecode::Jl => f.sign != f.overflow,
            Bytecode::Jle => f.zero || (f.sign != f.overflow),

            // Unsigned conditions
            Bytecode::Ja => !f.carry && !f.zero,
            Bytecode::Jae => !f.carry,
            Bytecode::Jb => f.carry,
            Bytecode::Jbe => f.carry || f.zero,

            // Other flag conditions
            Bytecode::Jc => f.carry,
            Bytecode::Jnc => !f.carry,
            Bytecode::Jo => f.overflow,
            Bytecode::Jno => !f.overflow,
            Bytecode::Js => f.sign,
            Bytecode::Jns => !f.sign,

            // Special conditions
            Bytecode::Jxcz => {
                let cx = self.registers.get(1).copied().unwrap_or(0);
                cx == 0
            }

            // For all other instructions that are not conditional jumps:
            _ => false,
        }
    }

    fn update_flags_add(&mut self, a: u32, b: u32, result: u32) {
        // Zero flag
        self.flags.zero = result == 0;

        // Sign flag (check MSB of result)
        self.flags.sign = (result as i32) < 0;

        // Carry flag: if unsigned addition overflows
        let (_, c) = a.overflowing_add(b);
        self.flags.carry = c;

        // Overflow flag: check signed overflow
        // Signed overflow occurs if (a and b have same sign) and (result differs in sign)
        let a_sign = (a as i32) < 0;
        let b_sign = (b as i32) < 0;
        let r_sign = (result as i32) < 0;
        self.flags.overflow = (a_sign == b_sign) && (a_sign != r_sign);
    }

    fn update_flags_sub(&mut self, a: u32, b: u32, result: u32) {
        // Zero flag
        self.flags.zero = result == 0;

        // Sign flag
        self.flags.sign = (result as i32) < 0;

        // Carry flag for subtraction: set if there's a borrow.
        // A borrow occurs if b > a in an unsigned sense.
        self.flags.carry = b > a;

        // Overflow flag: For subtraction, overflow occurs if the sign of a and b differ
        // and the sign of the result differs from a.
        let a_sign = (a as i32) < 0;
        let b_sign = (b as i32) < 0;
        let r_sign = (result as i32) < 0;

        self.flags.overflow = (a_sign != b_sign) && (r_sign != a_sign);
    }

    fn read_byte(&mut self, address: u32) -> Result<u8, ExecutionError> {
        self.valid_address(address)?;
        let cost = self.access(address, MemoryAccessDirection::LoadData);
        self.stats.cycles += cost;

        let byte_index = (address as usize) * 4;
        if byte_index + 1 > self.memory.len() * 4 {
            return Err(ExecutionError::InvalidMemoryLocation(address));
        }

        let mem_as_bytes = unsafe {
            std::slice::from_raw_parts(self.memory.as_ptr() as *const u8, self.memory.len() * 4)
        };
        Ok(mem_as_bytes[byte_index])
    }

    fn write_byte(&mut self, address: u32, value: u8) -> Result<(), ExecutionError> {
        self.valid_address(address)?;
        let cost = self.access(address, MemoryAccessDirection::Store);
        self.stats.cycles += cost;

        let byte_index = (address as usize) * 4;
        if byte_index + 1 > self.memory.len() * 4 {
            return Err(ExecutionError::InvalidMemoryLocation(address));
        }

        let mem_as_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                self.memory.as_mut_ptr() as *mut u8,
                self.memory.len() * 4,
            )
        };
        mem_as_bytes[byte_index] = value;
        Ok(())
    }

    fn update_flags_float(&mut self, a: f32, b: f32) {
        // Zero flag: a == b (but beware NaNs: if either is NaN, a == b is false)
        self.flags.zero = (a == b) && a.is_finite() && b.is_finite();

        // Sign flag: a < b
        self.flags.sign = a < b;

        // Overflow flag: Set if a or b is not finite (NaN or Infinity)
        self.flags.overflow = !(a.is_finite() && b.is_finite());

        // Carry flag: Clear it (no meaning in floats)
        self.flags.carry = false;
    }

    fn write_register(&mut self, reg: u32, value: u32) {
        if reg != 0 {
            self.registers[reg as usize] = value;
        }
    }

    fn read_register(&self, reg: u32) -> u32 {
        if reg != 0 {
            self.registers[reg as usize]
        } else {
            0
        }
    }

    fn run(&mut self, cycles: isize) -> Result<CpuStats, ExecutionError> {
        if self.halted {
            return Ok(self.stats);
        }

        let mut executed = 0;

        while (cycles < 0 || executed < cycles)
            && (self.instruction_pointer as usize) < self.memory.len()
        {
            let opcode = self.read_instruction(self.instruction_pointer)?;

            match opcode {
                Bytecode::Nop => {
                    if self.verbose {
                        println!("NOP");
                    }
                }
                Bytecode::LoadValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    if self.verbose {
                        println!("LOAD R{}, {}", reg, imm);
                    }

                    self.write_register(reg, imm);
                }
                Bytecode::LoadFromRegMemory => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let addr = self.read_register(reg2);

                    if self.verbose {
                        if self.addresses_as_integers {
                            println!("LOADREG R{}, R{}(@{})", reg1, reg2, addr);
                        } else {
                            println!("LOADREG R{}, R{}(@{:#02x})", reg1, reg2, addr);
                        }
                    }

                    let value = self.read_memory(addr)?;
                    self.write_register(reg1, value);
                }
                Bytecode::LoadMemory => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let addr = self.read_operand(self.instruction_pointer)?;

                    if self.verbose {
                        if self.addresses_as_integers {
                            println!("LOAD R{}, @{}", reg, addr);
                        } else {
                            println!("LOAD R{}, @{:#02x}", reg, addr);
                        }
                    }

                    let value = self.read_memory(addr)?;
                    self.write_register(reg, value);
                }
                Bytecode::LoadReg => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    if self.verbose {
                        println!("LOAD R{}, R{}", reg1, reg2);
                    }

                    self.write_register(reg1, self.read_register(reg2));
                }
                Bytecode::Store => {
                    let addr = self.read_operand(self.instruction_pointer)?;
                    let reg = self.read_operand(self.instruction_pointer)?;

                    if self.verbose {
                        if self.addresses_as_integers {
                            println!("STORE @{}, R{}", addr, reg);
                        } else {
                            println!("STORE @{:#02x}, R{}", addr, reg);
                        }
                    }

                    self.write_memory(addr, self.read_register(reg))?;
                }
                Bytecode::StoreAtReg => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let addr = self.read_register(reg1);

                    if self.verbose {
                        if self.addresses_as_integers {
                            println!("STOREREG R{}(@{}), R{}", reg1, addr, reg2);
                        } else {
                            println!("STOREREG R{}(@{:#02x}), R{}", reg1, addr, reg2);
                        }
                    }

                    self.write_memory(addr, self.read_register(reg2))?;
                }
                Bytecode::StoreValue => {
                    let addr = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    if self.verbose {
                        if self.addresses_as_integers {
                            println!("STORE @{}, {}", addr, imm);
                        } else {
                            println!("STORE @{:#02x}, {}", addr, imm);
                        }
                    }

                    self.write_memory(addr, imm)?;
                }
                Bytecode::StoreValueAtReg => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let addr = self.read_register(reg);

                    if self.verbose {
                        if self.addresses_as_integers {
                            println!("STOREREG R{}(@{}), {}", reg, addr, imm);
                        } else {
                            println!("STOREREG R{}(@{:#02x}), {}", reg, addr, imm);
                        }
                    }

                    self.write_memory(addr, imm)?;
                }
                Bytecode::PushValue => {
                    let imm = self.read_operand(self.instruction_pointer)?;

                    if self.verbose {
                        println!("PUSH {}", imm);
                    }

                    self.push_stack(imm)?;
                }
                Bytecode::PushReg => {
                    let reg = self.read_operand(self.instruction_pointer)?;

                    if self.verbose {
                        println!("PUSH R{}", reg);
                    }

                    let val = self.read_register(reg);

                    self.push_stack(val)?;
                }
                Bytecode::Pop => {
                    let reg = self.read_operand(self.instruction_pointer)?;

                    if self.verbose {
                        println!("POP R{}", reg);
                    }

                    let value = self.pop_stack()?;
                    self.write_register(reg, value);
                }
                Bytecode::And => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg1);
                    let b = self.read_register(reg2);
                    let result = a & b;

                    if self.verbose {
                        println!("AND R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.write_register(reg1, result);
                    self.update_flags_and(a, b, result);
                }
                Bytecode::AndValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg);
                    let b = imm;
                    let result = a & b;

                    if self.verbose {
                        println!("AND R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.write_register(reg, result);
                    self.update_flags_and(a, b, result);
                }
                Bytecode::Or => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg1);
                    let b = self.read_register(reg2);
                    let result = a | b;

                    if self.verbose {
                        println!("OR R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.write_register(reg1, result);
                    self.update_flags_or(a, b, result);
                }
                Bytecode::OrValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg);
                    let b = imm;
                    let result = a | b;

                    if self.verbose {
                        println!("OR R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.write_register(reg, result);
                    self.update_flags_or(a, b, result);
                }
                Bytecode::Xor => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg1);
                    let b = self.read_register(reg2);
                    let result = a ^ b;

                    if self.verbose {
                        println!("XOR R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.write_register(reg1, result);
                    self.update_flags_xor(a, b, result);
                }
                Bytecode::XorValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg);
                    let b = imm;
                    let result = a ^ b;

                    if self.verbose {
                        println!("XOR R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.write_register(reg, result);
                    self.update_flags_xor(a, b, result);
                }
                Bytecode::Not => {
                    let reg = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg);
                    let result = !a;

                    if self.verbose {
                        println!("NOT R{}({}) => {}", reg, a, result);
                    }

                    self.write_register(reg, result);
                    self.update_flags_not(a, result);
                }
                Bytecode::ShiftLeft => {
                    // shift R<reg1> by R<reg2> (logical left)
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let original = self.read_register(reg1);
                    let shift_count = self.read_register(reg2) & SHIFT_MASK;

                    let (result, carry_out) = self.shift_left(original, shift_count);
                    self.write_register(reg1, result);

                    // update flags
                    self.update_flags_shl(original, result, carry_out, shift_count);

                    if self.verbose {
                        println!(
                            "SHL R{}({:#x}), R{}({}) => {:#x}, CF={}",
                            reg1, original, reg2, shift_count, result, carry_out
                        );
                    }
                }
                Bytecode::ShiftLeftValue => {
                    // shift R<reg> by immediate (logical left)
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?; // shift amount

                    let original = self.read_register(reg1);
                    let shift_count = imm & SHIFT_MASK;

                    let (result, carry_out) = self.shift_left(original, shift_count);
                    self.write_register(reg1, result);

                    // update flags
                    self.update_flags_shl(original, result, carry_out, shift_count);

                    if self.verbose {
                        println!(
                            "SHL R{}({:#x}), #{} => {:#x}, CF={}",
                            reg1, original, shift_count, result, carry_out
                        );
                    }
                }
                Bytecode::ShiftRight => {
                    // shift R<reg1> by R<reg2> (logical right)
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let original = self.read_register(reg1);
                    let shift_count = self.read_register(reg2) & SHIFT_MASK;

                    let (result, carry_out) = self.shift_right(original, shift_count);
                    self.write_register(reg1, result);

                    // update flags
                    self.update_flags_shr(result, carry_out);

                    if self.verbose {
                        println!(
                            "SHR R{}({:#x}), R{}({}) => {:#x}, CF={}",
                            reg1, original, reg2, shift_count, result, carry_out
                        );
                    }
                }
                Bytecode::ShiftRightValue => {
                    // shift R<reg> by immediate (logical right)
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?; // shift amount

                    let original = self.read_register(reg1);
                    let shift_count = imm & SHIFT_MASK;

                    let (result, carry_out) = self.shift_right(original, shift_count);
                    self.write_register(reg1, result);

                    // update flags
                    self.update_flags_shr(result, carry_out);

                    if self.verbose {
                        println!(
                            "SHR R{}({:#x}), #{} => {:#x}, CF={}",
                            reg1, original, shift_count, result, carry_out
                        );
                    }
                }
                Bytecode::Add => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg1);
                    let b = self.read_register(reg2);
                    let result = a.wrapping_add(b);

                    if self.verbose {
                        println!("ADD R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.write_register(reg1, result);
                    self.update_flags_add(a, b, result);
                }
                Bytecode::AddValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg);
                    let b = imm;
                    let result = a.wrapping_add(b);

                    if self.verbose {
                        println!("ADD R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.write_register(reg, result);
                    self.update_flags_add(a, b, result);
                }
                Bytecode::FAdd => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = f32::from_bits(self.read_register(reg1));
                    let b = f32::from_bits(self.read_register(reg2));
                    let result = a + b;

                    if self.verbose {
                        println!("FADD R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.write_register(reg1, result.to_bits());
                    self.update_flags_float(a, b);
                }
                Bytecode::FAddValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = f32::from_bits(self.read_register(reg));
                    let b = f32::from_bits(imm);
                    let result = a + b;

                    if self.verbose {
                        println!("FADD R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.write_register(reg, result.to_bits());
                    self.update_flags_float(a, b);
                }
                Bytecode::Sub => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg1);
                    let b = self.read_register(reg2);
                    let result = a.wrapping_sub(b);

                    if self.verbose {
                        println!("SUB R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.write_register(reg1, result);
                    self.update_flags_sub(a, b, result);
                }
                Bytecode::SubValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg);
                    let b = imm;
                    let result = a.wrapping_sub(b);

                    if self.verbose {
                        println!("SUB R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.write_register(reg, result);
                    self.update_flags_sub(a, b, result);
                }
                Bytecode::FSub => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = f32::from_bits(self.read_register(reg1));
                    let b = f32::from_bits(self.read_register(reg2));
                    let result = a - b;

                    if self.verbose {
                        println!("FSUB R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.write_register(reg1, result.to_bits());
                    self.update_flags_float(a, b);
                }
                Bytecode::FSubValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = f32::from_bits(self.read_register(reg));
                    let b = f32::from_bits(imm);
                    let result = a - b;

                    if self.verbose {
                        println!("FSUB R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.write_register(reg, result.to_bits());
                    self.update_flags_float(a, b);
                }
                Bytecode::Mul => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg1);
                    let b = self.read_register(reg2);

                    let wide_result = (a as u64).wrapping_mul(b as u64);
                    let result = wide_result as u32;

                    if self.verbose {
                        println!("MUL R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.write_register(reg1, result);
                    self.update_flags_mul(a, b, wide_result);
                }
                Bytecode::MulValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg);
                    let b = imm;

                    let wide_result = (a as u64).wrapping_mul(b as u64);
                    let result = wide_result as u32;

                    if self.verbose {
                        println!("MUL R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.write_register(reg, result);
                    self.update_flags_mul(a, b, wide_result);
                }
                Bytecode::FMul => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = f32::from_bits(self.read_register(reg1));
                    let b = f32::from_bits(self.read_register(reg2));
                    let result = a * b;

                    if self.verbose {
                        println!("FMUL R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.write_register(reg1, result.to_bits());
                    self.update_flags_float(a, b);
                }
                Bytecode::FMulValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = f32::from_bits(self.read_register(reg));
                    let b = f32::from_bits(imm);
                    let result = a * b;

                    if self.verbose {
                        println!("FMUL R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.write_register(reg, result.to_bits());
                    self.update_flags_float(a, b);
                }
                Bytecode::Div => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg1);
                    let b = self.read_register(reg2);

                    if b == 0 {
                        return Err(ExecutionError::DivisionByZero);
                    }

                    let result = a.wrapping_div(b);

                    if self.verbose {
                        println!("DIV R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.write_register(reg1, result);
                    self.update_flags_div(a, b, result);
                }
                Bytecode::DivValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg);
                    let b = imm;

                    if b == 0 {
                        return Err(ExecutionError::DivisionByZero);
                    }

                    let result = a.wrapping_div(b);

                    if self.verbose {
                        println!("DIV R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.write_register(reg, result);
                    self.update_flags_div(a, b, result);
                }
                Bytecode::FDiv => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = f32::from_bits(self.read_register(reg1));
                    let b = f32::from_bits(self.read_register(reg2));

                    if b == 0.0 {
                        return Err(ExecutionError::DivisionByZero);
                    }

                    let result = a / b;

                    if self.verbose {
                        println!("FDIV R{}({}), R{}({}) => {}", reg1, a, reg2, b, result);
                    }

                    self.write_register(reg1, result.to_bits());
                    self.update_flags_float(a, b);
                }
                Bytecode::FDivValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = f32::from_bits(self.read_register(reg));
                    let b = f32::from_bits(imm);

                    if b == 0.0 {
                        return Err(ExecutionError::DivisionByZero);
                    }

                    let result = a / b;

                    if self.verbose {
                        println!("FDIV R{}({}), {} => {}", reg, a, b, result);
                    }

                    self.write_register(reg, result.to_bits());
                    self.update_flags_float(a, b);
                }
                Bytecode::LoadByte => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let addr = self.read_operand(self.instruction_pointer)?;

                    let byte = self.read_byte(addr)?;
                    self.write_register(reg, byte as u32);
                }

                Bytecode::StoreByte => {
                    let addr = self.read_operand(self.instruction_pointer)?;
                    let reg = self.read_operand(self.instruction_pointer)?;

                    let value = (self.read_register(reg) & 0xFF) as u8;
                    self.write_byte(addr, value)?;
                }
                Bytecode::Cmp => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg1);
                    let b = self.read_register(reg2);

                    let result = a.wrapping_sub(b);

                    if self.verbose {
                        println!(
                            "CMP R{}({}), R{}({}) => (flags updated for {} - {})",
                            reg1, a, reg2, b, a, b
                        );
                    }

                    self.update_flags_sub(a, b, result);
                }

                Bytecode::CmpValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = self.read_register(reg);
                    let b = imm;

                    let result = a.wrapping_sub(b);

                    if self.verbose {
                        println!(
                            "CMP R{}({}), {} => (flags updated for {} - {})",
                            reg, a, b, a, b
                        );
                    }

                    self.update_flags_sub(a, b, result);
                }
                Bytecode::FCmp => {
                    let reg1 = self.read_operand(self.instruction_pointer)?;
                    let reg2 = self.read_operand(self.instruction_pointer)?;

                    let a = f32::from_bits(self.read_register(reg1));
                    let b = f32::from_bits(self.read_register(reg2));

                    if self.verbose {
                        println!(
                            "FCMP R{}({}), R{}({}) => (flags updated for {} - {})",
                            reg1, a, reg2, b, a, b
                        );
                    }

                    self.update_flags_float(a, b);
                }
                Bytecode::FCmpValue => {
                    let reg = self.read_operand(self.instruction_pointer)?;
                    let imm = self.read_operand(self.instruction_pointer)?;

                    let a = f32::from_bits(self.read_register(reg));
                    let b = f32::from_bits(imm);

                    if self.verbose {
                        println!(
                            "FCMP R{}({}), {} => (flags updated for {} - {})",
                            reg, a, b, a, b
                        );
                    }

                    self.update_flags_float(a, b);
                }
                Bytecode::Jmp => {
                    let imm = self.read_operand(self.instruction_pointer)?;

                    if self.verbose {
                        println!("JMP {}", imm);
                    }
                    self.instruction_pointer = imm;
                }
                Bytecode::Je
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
                | Bytecode::Jxcz => {
                    let imm = self.read_operand(self.instruction_pointer)?;

                    if self.should_jump(opcode) {
                        if self.verbose {
                            println!("CONDITIONAL JUMP ({opcode}) to {imm}");
                        }
                        self.instruction_pointer = imm;
                    } else if self.verbose {
                        println!("CONDITIONAL JUMP ({opcode}) not taken");
                    }
                }
                Bytecode::Call => {
                    let target = self.read_operand(self.instruction_pointer)?;

                    if self.verbose {
                        if self.addresses_as_integers {
                            println!("CALL @{target}");
                        } else {
                            println!("CALL @{target:#02x}");
                        }
                    }

                    self.push_stack(self.instruction_pointer)?;

                    self.instruction_pointer = target;
                }
                Bytecode::Ret => {
                    let return_address = self.pop_stack()?;

                    if self.verbose {
                        if self.addresses_as_integers {
                            println!("RET to @{return_address}");
                        } else {
                            println!("RET to @{return_address:#02x}");
                        }
                    }

                    self.instruction_pointer = return_address;
                }
                Bytecode::Syscall => {
                    let code = self.read_register(1);
                    if self.verbose {
                        println!("SYSCALL code={}", code);
                    }

                    if let Some(mut host_io) = self.host_io.take() {
                        let return_value = host_io.syscall(code, self)?;
                        self.stats.cycles += return_value;
                        self.host_io.replace(host_io);
                    } else {
                        return Err(ExecutionError::NoHostIO);
                    }
                }
                Bytecode::Inspect => {
                    let addr = self.read_operand(self.instruction_pointer)?;

                    if self.addresses_as_integers {
                        println!("INSPECT @{} = {}", addr, self.read_memory(addr)?);
                    } else {
                        println!("INSPECT @{:#02x} = {}", addr, self.read_memory(addr)?);
                    }
                }
                Bytecode::Halt => {
                    self.set_halted(true);
                    if self.verbose {
                        println!("HALT");
                    }
                    break;
                }
            }

            self.stats.cycles += Self::update_cycles(opcode);

            executed += 1;
        }
        Ok(self.stats)
    }

    fn update_cycles(opcode: Bytecode) -> usize {
        match opcode {
            Bytecode::Nop
            | Bytecode::LoadFromRegMemory
            | Bytecode::StoreAtReg
            | Bytecode::StoreValueAtReg
            | Bytecode::LoadValue
            | Bytecode::LoadMemory
            | Bytecode::LoadReg
            | Bytecode::Store
            | Bytecode::StoreValue
            | Bytecode::PushValue
            | Bytecode::PushReg
            | Bytecode::Pop
            | Bytecode::And
            | Bytecode::AndValue
            | Bytecode::Or
            | Bytecode::OrValue
            | Bytecode::Xor
            | Bytecode::XorValue
            | Bytecode::Not
            | Bytecode::ShiftLeft
            | Bytecode::ShiftLeftValue
            | Bytecode::ShiftRight
            | Bytecode::ShiftRightValue
            | Bytecode::Add
            | Bytecode::AddValue
            | Bytecode::Sub
            | Bytecode::SubValue
            | Bytecode::Cmp
            | Bytecode::CmpValue
            | Bytecode::Jmp => 1,
            Bytecode::LoadByte
            | Bytecode::StoreByte
            | Bytecode::FCmp
            | Bytecode::FCmpValue
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
            | Bytecode::FSub
            | Bytecode::FSubValue
            | Bytecode::FAdd
            | Bytecode::FAddValue => 2,
            Bytecode::Mul | Bytecode::MulValue | Bytecode::FMul | Bytecode::FMulValue => 4,
            Bytecode::Ret => 5,
            Bytecode::Call => 25,
            Bytecode::Div | Bytecode::DivValue | Bytecode::FDiv | Bytecode::FDivValue => 27,
            Bytecode::Syscall | Bytecode::Inspect | Bytecode::Halt => 0,
        }
    }

    fn valid_address(&self, address: u32) -> Result<(), ExecutionError> {
        if (address as usize) < self.memory.len() {
            Ok(())
        } else {
            Err(ExecutionError::InvalidMemoryLocation(address))
        }
    }

    fn push_stack(&mut self, value: u32) -> Result<(), ExecutionError> {
        if self.stack_pointer == 0 {
            return Err(ExecutionError::StackOverflow);
        }

        self.write_memory(self.stack_pointer, value)?;
        self.stack_pointer -= 1;
        Ok(())
    }

    fn pop_stack(&mut self) -> Result<u32, ExecutionError> {
        if self.stack_pointer as usize >= self.memory.len() - 1 {
            return Err(ExecutionError::StackUnderflow);
        }

        self.stack_pointer += 1;
        self.read_memory(self.stack_pointer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_program(program: &[u32], memory_size: u32, registers: u8) -> NativeCpu<NullHostIO> {
        let mut cpu = NativeCpu::new(memory_size, registers, NullHostIO);
        cpu.load_protected_memory(0, program);
        cpu.execute(RunMode::Run).unwrap();
        cpu
    }

    #[test]
    fn test_nop() {
        let program = &[Bytecode::Nop as u32, Bytecode::Halt as u32];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.instruction_pointer, 2);
    }

    #[test]
    fn test_load_value() {
        let program = &[
            Bytecode::LoadValue as u32,
            1,  // R1
            42, // value
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[1], 42);
    }

    #[test]
    fn test_add() {
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            5, // R1 = 5
            Bytecode::LoadValue as u32,
            2,
            10, // R2 = 10
            Bytecode::Add as u32,
            1,
            2, // R1 = R1 + R2
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[1], 15);
        assert!(!cpu.flags.zero);
    }

    #[test]
    fn test_sub() {
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            10, // R1 = 10
            Bytecode::LoadValue as u32,
            2,
            5, // R2 = 5
            Bytecode::Sub as u32,
            1,
            2, // R1 = R1 - R2 (10-5 =5)
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[1], 5);
        assert!(!cpu.flags.zero);
        assert!(!cpu.flags.sign);
    }

    #[test]
    fn test_mul() {
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            6, // R1 = 6
            Bytecode::LoadValue as u32,
            2,
            7, // R2 = 7
            Bytecode::Mul as u32,
            1,
            2, // R1 = R1 * R2 (6*7=42)
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[1], 42);
    }

    #[test]
    fn test_div() {
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            42,
            Bytecode::LoadValue as u32,
            2,
            6,
            Bytecode::Div as u32,
            1,
            2, // R1 = R1 / R2 = 42/6 =7
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[1], 7);
    }

    #[test]
    fn test_write_r0() {
        let program = &[Bytecode::LoadValue as u32, 0, 42, Bytecode::Halt as u32];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[0], 0);
    }

    #[test]
    fn test_cmp_je() {
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            5,
            Bytecode::LoadValue as u32,
            2,
            5,
            Bytecode::Cmp as u32,
            1,
            2,
            Bytecode::Je as u32,
            15,
            Bytecode::LoadValue as u32,
            3,
            100, // If not equal, R3=100
            Bytecode::Halt as u32,
            // Jump target (index 15):
            Bytecode::LoadValue as u32,
            3,
            999,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        assert_eq!(cpu.get_registers()[3], 999);
    }

    #[test]
    fn test_stack_operations() {
        let program = &[
            Bytecode::PushValue as u32,
            42, // Push 42 on stack
            Bytecode::Pop as u32,
            1, // Pop into R1 => R1=42
            Bytecode::PushValue as u32,
            10, // push 10
            Bytecode::PushValue as u32,
            20, // push 20
            Bytecode::Pop as u32,
            2, // pop into R2 => R2=20
            Bytecode::Pop as u32,
            3, // pop into R3 => R3=10
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        let registers = cpu.get_registers();
        assert_eq!(registers[1], 42);
        assert_eq!(registers[2], 20);
        assert_eq!(registers[3], 10);
    }

    #[test]
    fn test_store_load_memory() {
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            123, // R1=123
            Bytecode::Store as u32,
            50,
            1, // memory[50]=R1=123
            Bytecode::LoadMemory as u32,
            2,
            50, // R2=memory[50] = 123
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        let registers = cpu.get_registers();
        assert_eq!(registers[2], 123);
        assert_eq!(cpu.get_memory()[50], 123);
    }

    #[derive(Debug)]
    struct TestHostIO {
        pub calls: Vec<(u32, u32, u32)>,
    }

    impl HostIO for TestHostIO {
        fn syscall(
            &mut self,
            code: u32,
            cpu: &mut NativeCpu<Self>,
        ) -> Result<usize, ExecutionError> {
            let arg1 = cpu.registers.get(2).copied().unwrap_or(0);
            let arg2 = cpu.registers.get(3).copied().unwrap_or(0);
            self.calls.push((code, arg1, arg2));
            Ok(0)
        }
    }

    fn run_program_with_host<IO: HostIO>(
        program: &[u32],
        memory_size: u32,
        registers: u8,
        host_io: IO,
    ) -> NativeCpu<IO> {
        let mut cpu = NativeCpu::new(memory_size, registers, host_io);
        cpu.load_protected_memory(0, program);
        cpu.execute(RunMode::Run).unwrap();
        cpu
    }

    #[test]
    fn test_syscall() {
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            1, // R1=1 (syscall code)
            Bytecode::LoadValue as u32,
            2,
            123, // R2=123
            Bytecode::LoadValue as u32,
            3,
            456, // R3=456
            Bytecode::Syscall as u32,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program_with_host(program, 128, 8, TestHostIO { calls: vec![] });

        assert_eq!(cpu.host_io.as_ref().unwrap().calls.len(), 1);
        assert_eq!(cpu.host_io.as_ref().unwrap().calls[0], (1, 123, 456));
    }

    #[test]
    fn test_load_store_byte() {
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            42, // R1=42
            Bytecode::StoreByte as u32,
            10,
            1, // memory[10*4]=42
            Bytecode::LoadByte as u32,
            2,
            10, // R2 = memory[10*4]
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 1024, 8);

        assert_eq!(cpu.get_registers()[2], 42);
    }

    #[test]
    fn test_load_float() {
        let f_val = 1.5f32.to_bits();
        let program = &[Bytecode::LoadValue as u32, 1, f_val, Bytecode::Halt as u32];

        let cpu = run_program(program, 128, 8);
        let result_bits = cpu.get_registers()[1];
        let result_float = f32::from_bits(result_bits);

        assert!((result_float - 1.5).abs() < 1e-7);
    }

    #[test]
    fn test_fadd() {
        let f1 = 1.5f32.to_bits();
        let f2 = 2.5f32.to_bits();
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            f1, // R1=1.5
            Bytecode::LoadValue as u32,
            2,
            f2, // R2=2.5
            Bytecode::FAdd as u32,
            1,
            2, // R1=R1+R2 =4.0
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 8);
        let result_float = f32::from_bits(cpu.get_registers()[1]);

        assert!((result_float - 4.0).abs() < 1e-7);
    }

    #[test]
    fn test_fsub() {
        let f5 = 5.0f32.to_bits();
        let f3 = 3.0f32.to_bits();
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            f5,
            Bytecode::LoadValue as u32,
            2,
            f3,
            Bytecode::FSub as u32,
            1,
            2,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 8);
        let result = f32::from_bits(cpu.get_registers()[1]);

        assert!((result - 2.0).abs() < 1e-7);
    }

    #[test]
    fn test_fmul() {
        let f2 = 2.0f32.to_bits();
        let f3_5 = 3.5f32.to_bits();
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            f2,
            Bytecode::LoadValue as u32,
            2,
            f3_5,
            Bytecode::FMul as u32,
            1,
            2,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 8);
        let result = f32::from_bits(cpu.get_registers()[1]);

        assert!((result - 7.0).abs() < 1e-7);
    }

    #[test]
    fn test_fdiv() {
        let f10 = 10.0f32.to_bits();
        let f2 = 2.0f32.to_bits();
        let program = &[
            Bytecode::LoadValue as u32,
            1,
            f10,
            Bytecode::LoadValue as u32,
            2,
            f2,
            Bytecode::FDiv as u32,
            1,
            2,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 8);
        let result = f32::from_bits(cpu.get_registers()[1]);

        assert!((result - 5.0).abs() < 1e-7);
    }

    #[test]
    fn test_shift_left_register() {
        // Program flow:
        // 1) Load R1 with 0x80000001  (has MSB set and LSB set)
        // 2) Load R2 with 1          (shift amount in register)
        // 3) ShiftLeft R1, R2       (R1 <<= R2)
        // 4) Halt
        //
        // Expected:
        //   - R1 becomes 0x00000002
        //   - carry flag set to true (the MSB "fell off")
        //   - sign flag = false (since 0x00000002 is positive)
        //   - zero flag = false
        //   - overflow flag = true, because shifting by 1 changed the MSB
        //     (x86-like semantics if shift == 1)

        let program = &[
            Bytecode::LoadValue as u32,
            1,
            0x80000001, // R1 = 0x80000001
            Bytecode::LoadValue as u32,
            2,
            1, // R2 = 1
            Bytecode::ShiftLeft as u32,
            1,
            2, // SHL R1, R1
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        let registers = cpu.get_registers();

        // Check R1 result
        assert_eq!(
            registers[1], 0x00000002,
            "R1 should have been shifted left by 1"
        );

        // Check flags
        assert!(!cpu.flags.zero, "0x2 is not zero");
        assert!(!cpu.flags.sign, "0x2 has no sign bit set");
        assert!(cpu.flags.carry, "Should carry out the top bit 1");
        assert!(
            cpu.flags.overflow,
            "Shifting MSB out with shift=1 => overflow set"
        );
    }

    #[test]
    fn test_shift_left_value() {
        // Program flow:
        // 1) Load R1 with 0x00000001
        // 2) ShiftLeftValue R1, 4  => R1 <<= 4
        // 3) Halt
        //
        // Expected:
        //   - R1 = 0x00000010
        //   - carry = false, no high bits fell off
        //   - sign = false, zero = false, overflow = false (shift=4 won't set OF on x86-like semantics)

        let program = &[
            Bytecode::LoadValue as u32,
            1,
            0x00000001,
            Bytecode::ShiftLeftValue as u32,
            1,
            4,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        let registers = cpu.get_registers();

        assert_eq!(registers[1], 0x10, "R1 = 1 << 4 should be 0x10");

        // Check flags
        assert!(!cpu.flags.zero);
        assert!(!cpu.flags.sign);
        assert!(!cpu.flags.carry, "No high bit was lost shifting 0x1 by 4");
        // shift=4 => overflow=0 in our design
        assert!(!cpu.flags.overflow);
    }

    #[test]
    fn test_shift_right_register() {
        // Program flow:
        // 1) Load R1 with 0x000000FF
        // 2) Load R2 with 4
        // 3) ShiftRight R1, R2 => R1 >>= 4 (logical right)
        // 4) Halt
        //
        // Expected:
        //   - R1 = 0x0000000F
        //   - carry = bit that was shifted out (lowest 4 bits were 1111, so carry = 1 after final shift)
        //   - zero = false, sign = false, overflow = false (typical logical right)

        let program = &[
            Bytecode::LoadValue as u32,
            1,
            0x000000FF,
            Bytecode::LoadValue as u32,
            2,
            4,
            Bytecode::ShiftRight as u32,
            1,
            2,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        let registers = cpu.get_registers();

        assert_eq!(registers[1], 0x0000000F);

        // Check flags
        assert!(!cpu.flags.zero);
        assert!(!cpu.flags.sign);
        assert!(cpu.flags.carry, "The last bit shifted out was 1");
        assert!(!cpu.flags.overflow);
    }

    #[test]
    fn test_shift_right_value_zero_flag() {
        // Program flow:
        // 1) Load R1 with 0x00000001
        // 2) ShiftRightValue R1, 1 => R1 >>= 1
        // 3) ShiftRightValue R1, 1 => R1 >>= 1 again
        //    after two shifts, R1 = 0x00000000 => zero flag set
        // 4) Halt
        //
        // Expected:
        //   - final R1 = 0
        //   - zero = true
        //   - carry = ?  (the bit shifted out each time)
        //        first shift => carry=1 (LSB=1)
        //        second shift => carry=0 (LSB=0 from 0x00000000)
        //   - We'll just check final carry from the second shift (should be false)

        let program = &[
            Bytecode::LoadValue as u32,
            1,
            0x1,
            // SHIFT RIGHT R1 by #1 => 0x1 -> 0x0 (carry=1)
            Bytecode::ShiftRightValue as u32,
            1,
            1,
            // SHIFT RIGHT R1 by #1 => 0x0 -> 0x0 (carry=0)
            Bytecode::ShiftRightValue as u32,
            1,
            1,
            Bytecode::Halt as u32,
        ];

        let cpu = run_program(program, 128, 4);
        let registers = cpu.get_registers();

        assert_eq!(registers[1], 0x0, "Final shift result is 0");
        assert!(cpu.flags.zero, "Result is zero");
        assert!(!cpu.flags.sign, "0 not negative");
        assert!(!cpu.flags.carry, "Second shift from 0 => carry=0");
        assert!(!cpu.flags.overflow);
    }
}
