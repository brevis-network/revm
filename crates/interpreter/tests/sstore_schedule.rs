//! The `SSTORE` cost/refund table, against the EIPs rather than against revm.
//!
//! `gas::calc::tests::sstore_table_matches_the_branch_chains` proves `table == the branch
//! chains it was generated from`. That is the right test for the refactor, but it is a
//! same-source comparison: it cannot tell you whether the *schedule* is right, and a mutation
//! confirms it -- changing London's `SSTORE_CLEARS_SCHEDULE` from 4800 to 4700 in
//! `sstore_refund` leaves that test green, while
//! `the_table_matches_the_eips_entry_by_entry` below goes red.
//!
//! So this file supplies the other half: a reference schedule written straight from the EIP
//! text, compared for every revision, every storage transition and both cold/warm. The
//! branch chains are upstream revm's and unchanged, so today this is a statement about
//! upstream; it earns its keep the next time this fork is rebased and someone edits them.
//!
//! Reference sources:
//!
//! * Frontier/Homestead .. Petersburg -- the original two-case rule (`Gsset` 20000 /
//!   `Gsreset` 5000, `Rsclear` 15000). Note that `CONSTANTINOPLE` also lands here: EIP-1283
//!   was reverted before it activated on mainnet, and revm folds `CONSTANTINOPLE` into
//!   `PETERSBURG` for `SSTORE` (see the `SpecId::CONSTANTINOPLE` doc comment).
//! * EIP-2200 (Istanbul) -- net gas metering, `SLOAD_GAS` 800, `SSTORE_RESET_GAS` 5000,
//!   `SSTORE_CLEARS_SCHEDULE` 15000.
//! * EIP-2929 (Berlin) -- `SLOAD_GAS` becomes `WARM_STORAGE_READ_COST` 100,
//!   `SSTORE_RESET_GAS` becomes `5000 - COLD_SLOAD_COST` = 2900, and a cold slot pays
//!   `COLD_SLOAD_COST` 2100 on top.
//! * EIP-3529 (London) -- `SSTORE_CLEARS_SCHEDULE` becomes
//!   `SSTORE_RESET(5000) - COLD_SLOAD_COST(2100) + ACCESS_LIST_STORAGE_KEY(1900)` = 4800.

use primitives::{hardfork::SpecId, U256};
use revm_interpreter::{
    gas::{self, SStoreStatus, SSTORE_GAS, SSTORE_SPEC, SSTORE_STATUS_COUNT},
    SStoreResult,
};

const ALL_SPECS: [SpecId; 21] = [
    SpecId::FRONTIER,
    SpecId::FRONTIER_THAWING,
    SpecId::HOMESTEAD,
    SpecId::DAO_FORK,
    SpecId::TANGERINE,
    SpecId::SPURIOUS_DRAGON,
    SpecId::BYZANTIUM,
    SpecId::CONSTANTINOPLE,
    SpecId::PETERSBURG,
    SpecId::ISTANBUL,
    SpecId::MUIR_GLACIER,
    SpecId::BERLIN,
    SpecId::LONDON,
    SpecId::ARROW_GLACIER,
    SpecId::GRAY_GLACIER,
    SpecId::MERGE,
    SpecId::SHANGHAI,
    SpecId::CANCUN,
    SpecId::PRAGUE,
    SpecId::OSAKA,
    SpecId::AMSTERDAM,
];

/// Total `SSTORE` gas and refund per the EIPs, written out independently of revm's chains.
///
/// Returns `(total_cost, refund)`, where `total_cost` includes the cold surcharge.
fn eip_reference(spec: SpecId, original: u64, present: u64, new: u64, is_cold: bool) -> (u64, i64) {
    let istanbul = spec as u8 >= SpecId::ISTANBUL as u8;
    let berlin = spec as u8 >= SpecId::BERLIN as u8;
    let london = spec as u8 >= SpecId::LONDON as u8;

    if !istanbul {
        // Frontier: two cases, and a clear refund keyed on `present`, not on `original`.
        let cost = if present == 0 && new != 0 {
            20_000
        } else {
            5_000
        };
        let refund = if present != 0 && new == 0 { 15_000 } else { 0 };
        return (cost, refund);
    }

    // EIP-2200, with the EIP-2929 / EIP-3529 substitutions.
    let sload_gas: u64 = if berlin { 100 } else { 800 };
    let sstore_reset_gas: u64 = if berlin { 2_900 } else { 5_000 };
    let sstore_set_gas: u64 = 20_000;
    let clears: i64 = if london { 4_800 } else { 15_000 };
    let cold: u64 = if berlin && is_cold { 2_100 } else { 0 };

    let mut cost;
    let mut refund: i64 = 0;

    if present == new {
        cost = sload_gas;
    } else if original == present {
        if original == 0 {
            cost = sstore_set_gas;
        } else {
            cost = sstore_reset_gas;
            if new == 0 {
                refund += clears;
            }
        }
    } else {
        cost = sload_gas;
        if original != 0 {
            if present == 0 {
                refund -= clears;
            } else if new == 0 {
                refund += clears;
            }
        }
        if original == new {
            if original == 0 {
                refund += (sstore_set_gas - sload_gas) as i64;
            } else {
                refund += (sstore_reset_gas - sload_gas) as i64;
            }
        }
    }
    cost += cold;
    (cost, refund)
}

fn vals(o: u64, p: u64, n: u64) -> SStoreResult {
    SStoreResult {
        original_value: U256::from(o),
        present_value: U256::from(p),
        new_value: U256::from(n),
    }
}

/// The whole table, against the EIPs, for every revision and every transition shape.
///
/// Four values (zero plus three distinct non-zero) is *exhaustive*, not a sample: both the
/// classification and the schedules only ever ask whether two of the three words are equal
/// and whether each is zero, so every reachable combination of those predicates over three
/// words is realised by some triple over `{0, 1, 2, 3}`.
#[test]
fn the_table_matches_the_eips_entry_by_entry() {
    for spec in ALL_SPECS {
        let spec_row = SSTORE_SPEC[spec as usize];
        for o in 0..4u64 {
            for p in 0..4u64 {
                for n in 0..4u64 {
                    let v = vals(o, p, n);
                    let entry = SSTORE_GAS[spec as usize][gas::sstore_status(&v) as usize];
                    for is_cold in [false, true] {
                        let (want_total, want_refund) = eip_reference(spec, o, p, n, is_cold);
                        // What the interpreter charges: the static part before the storage
                        // load, then the table's dynamic part plus the cold surcharge.
                        let got_total = spec_row.static_cost as u64
                            + entry.dyn_cost as u64
                            + if is_cold {
                                spec_row.cold_extra as u64
                            } else {
                                0
                            };
                        assert_eq!(
                            got_total, want_total,
                            "{spec:?} {o}->{p}->{n} cold={is_cold}: cost"
                        );
                        assert_eq!(
                            entry.refund as i64, want_refund,
                            "{spec:?} {o}->{p}->{n} cold={is_cold}: refund"
                        );
                    }
                }
            }
        }
    }
}

/// The named EIP-2200 examples, spelled out at Prague so a reader can check them against the
/// EIP's own table by eye. `(original, present, new) -> (gas, refund)` with a warm slot.
#[test]
fn the_eip_2200_worked_examples_at_prague() {
    // EIP-2200's table, with the EIP-2929/3529 substitutions applied
    // (SLOAD_GAS = 100, SSTORE_RESET_GAS = 2900, CLEARS = 4800).
    let cases: &[(u64, u64, u64, u64, i64)] = &[
        // original, present, new, total warm gas, refund
        (0, 0, 0, 100, 0),              // no-op on an untouched empty slot
        (0, 0, 1, 20_000, 0),           // set
        (0, 1, 0, 100, 19_900),         // added then deleted in the same tx
        (0, 1, 2, 100, 0),              // added then reassigned
        (1, 1, 1, 100, 0),              // no-op
        (1, 1, 0, 2_900, 4_800),        // delete   (SSTORE_RESET_GAS is 2900 from Berlin)
        (1, 1, 2, 2_900, 0),            // modify
        (1, 2, 0, 100, 4_800),          // modified then deleted
        (1, 0, 2, 100, -4_800),         // deleted then added
        (1, 0, 1, 100, -4_800 + 2_800), // deleted then restored
        (1, 2, 1, 100, 2_800),          // modified then restored
        (1, 2, 3, 100, 0),              // reassigned twice
    ];
    let spec = SpecId::PRAGUE;
    let spec_row = SSTORE_SPEC[spec as usize];
    for &(o, p, n, want_gas, want_refund) in cases {
        let v = vals(o, p, n);
        let entry = SSTORE_GAS[spec as usize][gas::sstore_status(&v) as usize];
        assert_eq!(
            spec_row.static_cost as u64 + entry.dyn_cost as u64,
            want_gas,
            "{o}->{p}->{n}: gas"
        );
        assert_eq!(entry.refund as i64, want_refund, "{o}->{p}->{n}: refund");
        // Cold adds exactly `COLD_SLOAD_COST` and nothing else.
        assert_eq!(spec_row.cold_extra, 2_100);
    }
}

/// The classification, independently of the schedule: each of the nine statuses is reached by
/// the transition its doc comment names, and by no other shape.
#[test]
fn the_nine_statuses_are_reached_by_exactly_the_documented_shapes() {
    use SStoreStatus::*;
    let expect = |o: u64, p: u64, n: u64| -> SStoreStatus {
        // `X`, `Y`, `Z` distinct and non-zero, per the enum's doc comments.
        if n == p {
            return Assigned;
        }
        match (o == 0, p == 0, n == 0, o == p, o == n) {
            (true, true, _, _, _) => Added,                         // 0 -> 0 -> Z
            (false, _, true, true, _) => Deleted,                   // X -> X -> 0
            (false, false, false, true, _) => Modified,             // X -> X -> Z
            (false, true, false, false, false) => DeletedAdded,     // X -> 0 -> Z
            (false, false, true, false, _) => ModifiedDeleted,      // X -> Y -> 0
            (false, true, false, false, true) => DeletedRestored,   // X -> 0 -> X
            (true, false, true, false, _) => AddedDeleted,          // 0 -> Y -> 0
            (false, false, false, false, true) => ModifiedRestored, // X -> Y -> X
            _ => Assigned,
        }
    };
    for o in 0..4u64 {
        for p in 0..4u64 {
            for n in 0..4u64 {
                assert_eq!(
                    gas::sstore_status(&vals(o, p, n)),
                    expect(o, p, n),
                    "{o}->{p}->{n}"
                );
            }
        }
    }
}

/// The two tables are indexed by `spec_id as usize` with no bound of their own, so the row
/// order has to be the discriminant order and the row count has to be the variant count.
///
/// The second assertion is the one that matters going forward: this is a fork that will be
/// rebased on upstream revm, and a new `SpecId` variant leaves both tables 21 rows long. The
/// lookups in `sstore_at` are ordinary (bounds-checked) indexing, so that is a panic on the
/// first `SSTORE` of the new fork rather than a silent mis-charge -- but nothing catches it
/// at compile time.
#[test]
fn the_tables_are_indexed_by_the_spec_discriminant() {
    for (i, spec) in ALL_SPECS.iter().enumerate() {
        assert_eq!(*spec as usize, i, "ALL_SPECS is not in discriminant order");
    }
    assert_eq!(SSTORE_GAS.len(), ALL_SPECS.len());
    assert_eq!(SSTORE_SPEC.len(), ALL_SPECS.len());
    assert_eq!(SSTORE_GAS[0].len(), SSTORE_STATUS_COUNT);
    // A 22nd hard fork would need a 22nd row.
    assert!(
        SpecId::try_from_u8(ALL_SPECS.len() as u8).is_none(),
        "a SpecId variant was added without extending SSTORE_GAS/SSTORE_SPEC"
    );
    assert!(SpecId::try_from_u8((ALL_SPECS.len() - 1) as u8).is_some());
}
