use crate::{interpreter_types::InputsTr, CallInput};
use primitives::{Address, MaybeAddress, U256};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Inputs for the interpreter that are used for execution of the call.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InputsImpl {
    /// Storage of this account address is being used.
    pub target_address: Address,
    /// Address of the bytecode that is being executed. This field is not used inside Interpreter but it is used
    /// by dependent projects that would need to know the address of the bytecode.
    ///
    /// [`MaybeAddress`] rather than `Option<Address>` so the 20 bytes land somewhere the
    /// compiler knows is 8-aligned; see there.
    pub bytecode_address: MaybeAddress,
    /// Address of the caller of the call.
    pub caller_address: Address,
    /// Input data for the call.
    pub input: CallInput,
    /// Value of the call.
    pub call_value: U256,
}

impl InputsTr for InputsImpl {
    /// `Address` is `[u8; 20]` with alignment 1, so reading this field as a *value* is 20
    /// `lbu` plus the shift/or chain that reassembles them and the stores that put them back
    /// down again -- measured at 62 retired instructions per `SSTORE` on mainnet block
    /// 24006677, and the same per `SLOAD` and per `LOG`. Copying through
    /// `copy_address_bytes` states the alignment once and moves three words; the destination
    /// is forced to 8 so the check it makes cannot fall to the byte path.
    #[inline(always)]
    fn target_address(&self) -> Address {
        #[repr(align(8))]
        struct Aligned(core::mem::MaybeUninit<Address>);
        let mut out = Aligned(core::mem::MaybeUninit::uninit());
        // SAFETY: `out.0` is 20 writable bytes that do not overlap the field, and
        // `copy_address_bytes` writes all 20 of them, so `assume_init` sees a whole `Address`.
        unsafe {
            primitives::copy_address_bytes(
                out.0.as_mut_ptr().cast::<u8>(),
                self.target_address.0.as_ptr(),
            );
            out.0.assume_init()
        }
    }

    fn caller_address(&self) -> Address {
        self.caller_address
    }

    fn bytecode_address(&self) -> Option<&Address> {
        self.bytecode_address.get()
    }

    fn input(&self) -> &CallInput {
        &self.input
    }

    fn call_value(&self) -> U256 {
        self.call_value
    }
}
